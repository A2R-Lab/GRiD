"""Generate/run a private one-block local-tensor correctness probe. No timing.

Invoke --execute only through the correctness coordinator. Production emitters,
public scratch contracts, receipts and shared compilation caches are untouched.
Without --full-chain only the four ID second-order blocks move to the GPU.
With --full-chain the private kernel also computes mass, bias, equilibrated
Cholesky, acceleration, gradient and contraction from q/qd/u on the GPU.
"""
from __future__ import annotations

import argparse
import ast
import inspect
import json
import os
from pathlib import Path
import subprocess
import textwrap

import numpy as np

from tools import cpu_local_chain as cpu
from tools.run_cpu_local_chain import sha, metrics


def literal(value):
    value = format(float(value), '.17g')
    if '.' not in value and 'e' not in value: value += '.0'
    return value + 'f'


def expression(node, variables):
    if isinstance(node, ast.Constant) and isinstance(node.value, (float, int)):
        return literal(node.value)
    if isinstance(node, ast.Name): return variables[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return '(-' + expression(node.operand, variables) + ')'
    if isinstance(node, ast.BinOp):
        a, b = expression(node.left, variables), expression(node.right, variables)
        if isinstance(node.op, ast.Pow): return f'powf({a},{b})'
        op = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/'}[type(node.op)]
        return f'({a}{op}{b})'
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ('sin','cos','sqrt','tan'):
        return node.func.id+'f('+','.join(expression(a, variables) for a in node.args)+')'
    raise ValueError('unsupported symbolic transform expression')


def generate(model, full_chain=False):
    r = model.robot; nb = r.get_num_bodies(); n = model.nv
    if getattr(model.reference,'use_joint_dynamics',False):
        raise ValueError('joint-dynamics bias is outside this probe')
    # Validate candidate scope before generation; no oracle dynamics involved.
    probe = np.zeros(r.get_num_pos()); probe[6] = 1
    cpu.tree_data(model, probe)
    parent=[]; slot=[]; dim=[]; alpha=[]; subspaces=[]; inertias=[]; functions=[]
    for i in range(nb):
        joint=r.get_joint_by_id(i); vi=cpu.indices(r.get_joint_index_v(i)); qi=cpu.indices(r.get_joint_index_q(i))
        parent.append(r.get_parent_id(i)); slot.extend(vi+[-1]*(6-len(vi))); dim.append(len(vi))
        a = float(joint.get_mimic_multiplier()) if getattr(joint,'is_mimic',False) else 1.
        off = float(joint.get_mimic_offset()) if getattr(joint,'is_mimic',False) else 0.
        alpha.append(a)
        s=np.zeros((6,6),dtype=np.float32); s[:len(vi)]=np.asarray(r.get_S_by_id(i)).reshape(6,len(vi)).T
        subspaces.extend(s.reshape(-1)); inertias.extend(np.asarray(r.get_Imat_by_id(i),dtype=np.float32).reshape(-1))
        fn=ast.parse(textwrap.dedent(inspect.getsource(r.get_Xmat_Func_by_id(i)))).body[0]
        variables={}
        if len(qi)==1: variables[fn.args.args[0].arg]=f'({literal(a)}*q[{qi[0]}]+{literal(off)})'
        else:
            unpack=fn.body[0]
            if not isinstance(unpack,ast.Assign) or not isinstance(unpack.targets[0],(ast.List,ast.Tuple)):
                raise ValueError('expected positional root transform arguments')
            variables={v.id:f'q[{index}]' for v,index in zip(unpack.targets[0].elts,qi,strict=True)}
        value=fn.body[-1].value
        if not isinstance(value,ast.Call) or value.func.id!='array': raise ValueError('expected matrix formula')
        entries=[v for row in value.args[0].elts for v in row.elts]
        if len(entries)!=36: raise ValueError('expected six-by-six spatial transform')
        lines=[f'out[{k}]={expression(v,variables)};' for k,v in enumerate(entries)]
        functions.append(f'case {i}: {{'+''.join(lines)+'break;}')
    def array(name, values, integer=False):
        kind='int' if integer else 'float'
        return f'__device__ __constant__ {kind} {name}[]={{'+','.join(map(str,values) if integer else map(literal,values))+'};\n'
    head='#include <cuda_runtime.h>\n#include <cmath>\n'
    head+=f'constexpr int NB={nb},NV={n},NQ={r.get_num_pos()};\n'
    for name, values in [('PARENT',parent),('SLOT',slot),('DIM',dim)]: head+=array(name,values,True)
    for name, values in [('ALPHA',alpha),('S',subspaces),('INERTIA',inertias)]: head+=array(name,values)
    if full_chain: head+=array('ARMATURE',np.asarray(model.reference.armature,dtype=np.float32))
    head+='__device__ void evaluate_transform(int body,const float* q,float* out){switch(body){'+''.join(functions)+'}}\n'
    template='local_chain_probe.cuh' if full_chain else 'local_tensor_probe.cuh'
    return head+Path(__file__).with_name(template).read_text()


def run_admitted(command, log_path):
    from test.compile_sched import mem_available_kb, session_tree_rss_kb
    from test.resource_admission import MemoryAdmission
    admission=MemoryAdmission(floor_kb=14*1024**2,available=mem_available_kb,
                              rss=session_tree_rss_kb,log_path=log_path.with_suffix('.memory.jsonl'))
    if not admission.reserve('probe',int(16*1.3*1024**2)):
        raise RuntimeError(admission.refusal)
    with log_path.open('x') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        admission.attach('probe',process.pid)
        while process.poll() is None:
            admission.pressure()
            try: process.wait(timeout=1)
            except subprocess.TimeoutExpired: pass
        admission.release('probe')
    if process.returncode: raise subprocess.CalledProcessError(process.returncode,command)
    if admission.blocked: raise RuntimeError(admission.blocked)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--cases',nargs='+',help='robot/sample; omit for every saved input')
    parser.add_argument('--arch',default='sm_120')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--full-chain',action='store_true',help='also compute bias, mass, solves, gradients and contraction on GPU')
    args=parser.parse_args()
    if args.arch!='sm_120': raise ValueError('only the scoped local primary probe is authorized')
    if not args.cases:
        args.cases=[p.stem.replace('-','/',1) for p in sorted(args.captures.glob('*.npz'))]
    if not args.cases or len(args.cases)!=len(set(args.cases)):
        raise ValueError('empty or duplicate input scope')
    from test.cuda_equivalents import test_cuda_second_order_fallback as t
    args.out.mkdir(parents=True,exist_ok=False)
    meta=dict(source_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              source_status=subprocess.check_output(['git','status','--short'],text=True).strip(),
              scripts={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('local_tensor_probe.cuh'),
                       Path(__file__).with_name('local_chain_probe.cuh'),Path(cpu.__file__))},
              cases=args.cases,arch=args.arch,
              full_chain=args.full_chain,
              limitations=[('Entire dynamics chain on GPU; parsed model constants and q/qd/u uploaded' if args.full_chain
                            else 'ID tensors only on GPU; CPU acceleration, gradient, solve and contraction'),
                           'private emitter, not production integration or performance evidence'])
    if args.execute:
        meta['nvcc']=subprocess.check_output(['nvcc','--version'],text=True)
    (args.out/'provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
    models={}; failures=[]
    for case in args.cases:
        robot,sample=case.split('/',1); path=args.captures/f'{robot}-{sample}.npz'
        if robot not in models:
            spec=t._robot_spec(robot,'floating'); model=t.build_project_adapter(spec,t.resolve_robot_spec(spec),base_mode='floating')
            models[robot]=model
            if 'peers' not in meta:
                peer_root=Path(inspect.getfile(type(model.reference))).resolve().parent.parent
                meta['peers']={}
                for peer in ('RBDReference','URDFParser','GLASS'):
                    peer_path=peer_root/peer
                    if subprocess.check_output(['git','-C',str(peer_path),'status','--short'],text=True).strip():
                        raise ValueError('dirty study peer')
                    meta['peers'][peer]=subprocess.check_output(['git','-C',str(peer_path),'rev-parse','HEAD'],text=True).strip()
                (args.out/'provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
            source=args.out/f'{robot}.cu'; source.write_text(generate(model,args.full_chain))
            if args.execute:
                command=['nvcc','-O2','-std=c++17',f'-arch={args.arch}','-o',str(args.out/robot),str(source)]
                (args.out/f'{robot}.compile-command.json').write_text(json.dumps(command)+'\n')
                run_admitted(command,args.out/f'{robot}.compile.log')
        if not args.execute: continue
        model=models[robot]; n=model.nv
        with np.load(path) as z: q,qd,u=[np.asarray(z[k],dtype=np.float32) for k in ('q','qd','u')]
        candidate=cpu.chain(model,q,qd,u)
        source=args.out/f'{robot}-{sample}.input.bin'
        np.concatenate((q,qd,u if args.full_chain else candidate['qdd'])).astype(np.float32).tofile(source)
        expected=t._flatten_second_order_tensors(model.fdsva_so(q.astype(np.float64),qd.astype(np.float64),u.astype(np.float64)))
        outputs=[]; row=dict(case=case,capture_sha256=sha(path),results={})
        for index,threads in enumerate((1,32,128,32)):
            dest=args.out/f'{robot}-{sample}.{index}.output.bin'
            run_admitted([str(args.out/robot),str(source),str(dest),str(threads)],
                         args.out/f'{robot}-{sample}.{index}.log')
            tensors=np.fromfile(dest,dtype=np.float32).reshape(4,n,n,n)
            outputs.append(tensors)
            out=tensors if args.full_chain else cpu.contract(cpu.factor(candidate['matrix']),candidate['grad'],tensors)
            row['results'][str(index)]=dict(threads=threads,**metrics(t._flatten_second_order_tensors(out),expected,robot,t))
        row['bit_identical_threads_and_repeat']=all(outputs[0].tobytes()==v.tobytes() for v in outputs[1:])
        baseline=candidate['output'] if args.full_chain else (*candidate['tensors'],candidate['dm'])
        row['gpu_cpu_max']=float(np.max(np.abs(outputs[0]-np.asarray(baseline))))
        with (args.out/'results.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
        print(row,flush=True)
        if not row['bit_identical_threads_and_repeat'] or not all(v['gate'] for v in row['results'].values()):
            failures.append(case)
            break  # A failing product candidate stops GPU expansion.
    (args.out/'summary.json').write_text(json.dumps(dict(executed=args.execute,failures=failures),indent=2)+'\n')
    return 2 if failures else 0


if __name__=='__main__': raise SystemExit(main())
