"""Run the production row-copy helper with a CPU CUDA transport double."""
from pathlib import Path
import subprocess


def test_floating_output_drops_padding_without_overrunning(tmp_path):
    (tmp_path/'cuda_runtime_api.h').write_text('''
#include <cstddef>
#include <cstring>
using cudaError_t=int;
enum cudaMemcpyKind {cudaMemcpyDeviceToHost,cudaMemcpyDeviceToDevice};
static int failure=0;
inline int cudaMemcpy2D(void *d,size_t dp,const void *s,size_t sp,size_t w,size_t h,cudaMemcpyKind) {
 if(failure)return failure;
 for(size_t i=0;i<h;++i) std::memcpy((char*)d+i*dp,(const char*)s+i*sp,w);
 return 0;
}
''')
    source = tmp_path/'test.cpp'
    source.write_text('''
#include "consumer_copy.h"
#include <cassert>
int main() {
 float padded[]={1,2,3,999,4,5,6,999}, out[8]={-1,0,0,0,0,0,0,-1};
 for(auto kind:{cudaMemcpyDeviceToHost,cudaMemcpyDeviceToDevice}) {
  assert(!consumer_copy_dense(out+1,3,padded,4,2,kind));
  for(int i=0;i<6;++i)assert(out[i+1]==float(i+1));
  assert(out[0]==-1 && out[7]==-1);
 }
 failure=7;
 assert(consumer_copy_dense(out+1,3,padded,4,2,cudaMemcpyDeviceToHost)==7);
}
'''.replace('#include <cassert>', '#include <cassert>\n#include <initializer_list>'))
    include = Path(__file__).parent/'release'
    exe = tmp_path/'test'
    subprocess.run(['g++','-std=c++17','-I'+str(tmp_path),'-I'+str(include),str(source),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
