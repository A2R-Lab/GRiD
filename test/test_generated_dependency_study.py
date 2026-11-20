"""CPU source-inventory/parser safeguards; integration data is kept separately."""
import pytest

from tools.generated_dependency_study import definition_inventory, mask_cpp, mutate_function


def header(body='if (x) { x(); }'):
    return '\n'.join('template <typename T>\n__device__ void ' + name + '(T* x) {' + body + '}\n'
                     for name in ('inverse_dynamics_inner', 'inverse_dynamics_inner_vaf'))


def test_extract_definitions_ignores_comments_strings_and_calls():
    text = '// void inverse_dynamics_inner() {}\n' + header('const char* s="}"; /* } */ if (x) { x(); }')
    text += '\nvoid wrapper() { inverse_dynamics_inner<float>(nullptr); }'
    result = definition_inventory(text)
    assert all(len(v) == 1 for v in result.values())
    assert result['inverse_dynamics_inner'][0]['start_line'] == 3
    assert definition_inventory(header('x();'))['inverse_dynamics_inner'][0]['sha256'] != result['inverse_dynamics_inner'][0]['sha256']


def test_multiple_overloads_are_all_accounted_for():
    result = definition_inventory(header() + header('x(1);'))
    assert all(len(v) == 2 for v in result.values())
    assert all(v[0]['sha256'] != v[1]['sha256'] for v in result.values())


@pytest.mark.parametrize('text', ['', 'void inverse_dynamics_inner() {', 'void inverse_dynamics_inner('])
def test_missing_or_unbalanced_definition_rejected(text):
    with pytest.raises(ValueError): definition_inventory(text)


def test_raw_strings_are_explicitly_unresolved():
    with pytest.raises(ValueError, match='raw'): mask_cpp('auto s = R"(x)";')


def test_mutation_is_function_scoped_and_anchor_checked():
    src = 'def first():\n    print("x")\n\ndef second():\n    print("x")\n'
    actual = mutate_function(src, 'first', '"x"', '"y"')
    assert actual == src.replace('"x"', '"y"', 1)
    with pytest.raises(ValueError, match='unique'): mutate_function(src, 'first', 'missing', 'new')


def test_prototypes_do_not_count_as_definitions():
    result = definition_inventory('void inverse_dynamics_inner(int x);\n' + header())
    assert len(result['inverse_dynamics_inner']) == 1
