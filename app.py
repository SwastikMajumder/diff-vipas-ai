import copy
import ast
import streamlit as st

class TreeNode:
    def __init__(self, name, children=None):
        self.name = name
        self.children = children or []
def tree_form(tabbed_strings):
    lines = tabbed_strings.split("\n")
    root = TreeNode("Root") 
    current_level_nodes = {0: root}
    stack = [root]
    for line in lines:
        level = line.count(' ') 
        node_name = line.strip() 
        node = TreeNode(node_name)
        while len(stack) > level + 1:
            stack.pop()
        parent_node = stack[-1]
        parent_node.children.append(node)
        current_level_nodes[level] = node
        stack.append(node)
    return root.children[0] 
def str_form(node):
    def recursive_str(node, depth=0):
        result = "{}{}".format(' ' * depth, node.name) 
        for child in node.children:
            result += "\n" + recursive_str(child, depth + 1) 
        return result
    return recursive_str(node)
def apply_individual_formula_on_given_equation(equation, formula_lhs, formula_rhs, do_only_arithmetic=False, structure_satisfy=False):
    variable_list = {}
    def node_type(s):
        if s[:2] == "f_":
            return s
        else:
            return s[:2]
    def does_given_equation_satisfy_forumla_lhs_structure(equation, formula_lhs):
        nonlocal variable_list
        if node_type(formula_lhs.name) in {"u_", "p_"}: 
            if formula_lhs.name in variable_list.keys(): 
                return str_form(variable_list[formula_lhs.name]) == str_form(equation) 
            else: 
                if node_type(formula_lhs.name) == "p_" and "v_" in str_form(equation): 
                    return False
                variable_list[formula_lhs.name] = copy.deepcopy(equation)
                return True
        if equation.name != formula_lhs.name or len(equation.children) != len(formula_lhs.children): 
            return False
        for i in range(len(equation.children)): 
            if does_given_equation_satisfy_forumla_lhs_structure(equation.children[i], formula_lhs.children[i]) is False:
                return False
        return True
    if structure_satisfy:
      return does_given_equation_satisfy_forumla_lhs_structure(equation, formula_lhs)
    def formula_apply_root(formula):
        nonlocal variable_list
        if formula.name in variable_list.keys():
            return variable_list[formula.name] 
        data_to_return = TreeNode(formula.name, None) 
        for child in formula.children:
            data_to_return.children.append(formula_apply_root(copy.deepcopy(child))) 
        return data_to_return
    count_target_node = 1
    def formula_apply_various_sub_equation(equation, formula_lhs, formula_rhs, do_only_arithmetic):
        nonlocal variable_list
        nonlocal count_target_node
        data_to_return = TreeNode(equation.name, children=[])
        variable_list = {}
        if do_only_arithmetic == False:
            if does_given_equation_satisfy_forumla_lhs_structure(equation, copy.deepcopy(formula_lhs)) is True: 
                count_target_node -= 1
                if count_target_node == 0: 
                    return formula_apply_root(copy.deepcopy(formula_rhs)) 
        else: 
            if len(equation.children) == 2 and all(node_type(item.name) == "d_" for item in equation.children): 
                x = []
                for item in equation.children:
                    x.append(int(item.name[2:])) 
                if equation.name == "f_add":
                    count_target_node -= 1
                    if count_target_node == 0: 
                        return TreeNode("d_" + str(sum(x))) 
                elif equation.name == "f_mul":
                    count_target_node -= 1
                    if count_target_node == 0:
                        p = 1
                        for item in x:
                            p *= item 
                        return TreeNode("d_" + str(p))
                elif equation.name == "f_pow" and x[1]>=2: 
                    count_target_node -= 1
                    if count_target_node == 0:
                        return TreeNode("d_"+str(int(x[0]**x[1])))
                elif equation.name == "f_sub":
                    count_target_node -= 1
                    if count_target_node == 0: 
                        return TreeNode("d_" + str(x[0]-x[1]))
                elif equation.name == "f_div" and int(x[0]/x[1]) == x[0]/x[1]:
                    count_target_node -= 1
                    if count_target_node == 0: 
                        return TreeNode("d_" + str(int(x[0]/x[1])))
        if node_type(equation.name) in {"d_", "v_"}: 
            return equation
        for child in equation.children: 
            data_to_return.children.append(formula_apply_various_sub_equation(copy.deepcopy(child), formula_lhs, formula_rhs, do_only_arithmetic))
        return data_to_return
    cn = 0
    def count_nodes(equation):
        nonlocal cn
        cn += 1
        for child in equation.children:
            count_nodes(child)
    transformed_equation_list = []
    count_nodes(equation)
    for i in range(1, cn + 1): 
        count_target_node = i
        orig_len = len(transformed_equation_list)
        tmp = formula_apply_various_sub_equation(equation, formula_lhs, formula_rhs, do_only_arithmetic)
        if str_form(tmp) != str_form(equation): 
            transformed_equation_list.append(str_form(tmp)) 
    return transformed_equation_list
def generate_transformation(equation, file_name):
    input_f, output_f = return_formula_file(file_name) 
    transformed_equation_list = []
    for i in range(len(input_f)): 
        transformed_equation_list += apply_individual_formula_on_given_equation(tree_form(copy.deepcopy(equation)), copy.deepcopy(input_f[i]), copy.deepcopy(output_f[i]))
    return list(set(transformed_equation_list))
def generate_arithmetical_transformation(equation):
    transformed_equation_list = []
    transformed_equation_list += apply_individual_formula_on_given_equation(tree_form(equation), None, None, True) 
    return list(set(transformed_equation_list))
algebraic_formula =\
"""f_add
 u_0
 d_0

u_0

f_add
 d_0
 u_0

u_0

f_pow
 u_0
 d_-1

f_div
 d_1
 u_0

f_mul
 u_0
 d_0

d_0

f_mul
 d_0
 u_0

d_0

f_mul
 d_1
 u_0

u_0

f_mul
 u_0
 d_1

u_0

f_pow
 u_0
 d_1

u_0"""
diff_formula = \
"""f_dif
 f_cos
  u_0

f_mul
 f_dif
  u_0
 f_mul
  d_-1
  f_sin
   u_0

f_dif
 f_mul
  u_0
  u_1

f_add
 f_mul
  u_0
  f_dif
   u_1
 f_mul
  u_1
  f_dif
   u_0

f_dif
 f_sin
  u_0

f_mul
 f_dif
  u_0
 f_cos
  u_0

f_dif
 f_pow
  u_0
  p_0

f_mul
 f_pow
  u_0
  f_sub
   p_0
   d_1
 f_mul
  p_0
  f_dif
   u_0

f_dif
 v_0

d_1

f_dif
 p_0

d_0

f_dif
 f_add
  u_0
  u_1

f_add
 f_dif
  u_0
 f_dif
  u_1

f_dif
 f_div
  u_0
  u_1

f_div
 f_sub
  f_mul
   u_1
   f_dif
    u_0
  f_mul
   u_0
   f_dif
    u_1
 f_pow
  u_1
  d_2"""
def return_formula_file(file_name):
    global algebraic_formula
    global diff_formula
    content = None
    if file_name == "algebraic_formula.txt":
        content = algebraic_formula
    elif file_name == "diff_formula.txt":
        content = diff_formula
    x = content.split("\n\n")
    input_f = [x[i] for i in range(0, len(x), 2)] 
    output_f = [x[i] for i in range(1, len(x), 2)]
    input_f = [tree_form(item) for item in input_f] 
    output_f = [tree_form(item) for item in output_f]
    return [input_f, output_f]
def main_search(equation, file):
    while True:
        add_equation_to_log(string_equation(equation))
        tmp = generate_transformation(equation, file)+generate_arithmetical_transformation(equation)
        for item in tmp:
            if "f_dif" not in item:
                return item
        equation = tmp[0]
def post_search(equation, file):
    while True:
        add_equation_to_log(string_equation(equation))
        tmp = generate_transformation(equation, file)+generate_arithmetical_transformation(equation)
        if tmp == []:
            return equation
        equation = tmp[0]

def generate_transformation_2(equation, file_name):
    input_f, output_f = return_formula_file_2(file_name) 
    transformed_equation_list = []
    for i in range(len(input_f)): 
        transformed_equation_list += apply_individual_formula_on_given_equation(tree_form(copy.deepcopy(equation)), copy.deepcopy(input_f[i]), copy.deepcopy(output_f[i]))
    return list(set(transformed_equation_list)) 

def return_formula_file_2(file_name):
    content = None
    f_1 = \
"""f_mul
 u_0
 f_pow
  u_1
  d_-1

f_div
 u_0
 u_1

f_mul
 f_pow
  u_1
  d_-1
 u_0

f_div
 u_0
 u_1

f_mul
 d_1
 u_0

u_0"""
    f_2 = \
"""f_div
 u_0
 u_1

f_mul
 u_0
 f_pow
  u_1
  d_-1

f_pow
 f_mul
  u_0
  u_1
 p_0

f_mul
 f_pow
  u_0
  p_0
 f_pow
  u_1
  p_0"""
    if file_name == "formula-list-6/convert_division_2.txt":
        content = f_1
    elif file_name == "formula-list-6/convert_division.txt":
        content = f_2
    x = content.split("\n\n")
    input_f = [x[i] for i in range(0, len(x), 2)] 
    output_f = [x[i] for i in range(1, len(x), 2)]
    input_f = [tree_form(item) for item in input_f] 
    output_f = [tree_form(item) for item in output_f]
    return [input_f, output_f] 

def search(equation, depth, file_list):
    final = []
    def search_helper(equation, depth, file_list, auto_arithmetic=True, visited=None):
        if depth == 0: 
            return None
        if visited is None:
            visited = set()
        final.append(equation)

        if equation in visited:
            return None
        visited.add(equation)
        output =[]
        if file_list[0]:
          output += generate_transformation_2(equation, file_list[0])
        if auto_arithmetic:
          output += generate_arithmetical_transformation(equation)
        if len(output) > 0:
          output = [output[0]]
        else:
          if file_list[1]:
            output += generate_transformation_2(equation, file_list[1])
          if not auto_arithmetic:
            output += generate_arithmetical_transformation(equation)
          if file_list[2] and len(output) == 0:
              output += generate_transformation_2(equation, file_list[2])
        for i in range(len(output)):
            search_helper(output[i], depth-1, file_list, auto_arithmetic, visited) 
    search_helper(equation, depth, file_list)
    return final

def string_equation_helper(equation_tree):
    if equation_tree.children == []:
        return equation_tree.name 
    s = "(" 
    if len(equation_tree.children) == 1:
        s = equation_tree.name[2:] + s
    sign = {"f_add": "+", "f_mul": "*", "f_pow": "^", "f_poly": ",", "f_div": "/", "f_int": ",", "f_sub": "-", "f_dif": "?", "f_sin": "?", "f_cos": "?", "f_tan": "?", "f_eq": "=", "f_sqt": "?"} 
    for child in equation_tree.children:
        s+= string_equation_helper(copy.deepcopy(child)) + sign[equation_tree.name]
    s = s[:-1] + ")"
    return s

def string_equation(eq): 
    eq = eq.replace("v_0", "x")
    eq = eq.replace("v_1", "y")
    eq = eq.replace("v_2", "z")
    eq = eq.replace("d_", "")

    return string_equation_helper(tree_form(eq))

def replace(equation, find, r):
  if str_form(equation) == str_form(find):
    return r
  col = TreeNode(equation.name, [])
  for child in equation.children:
    col.children.append(replace(child, find, r))
  return col

def flatten_tree(node):
    if not node.children:
        return node
    if node.name in ("f_add", "f_mul"):
        merged_children = []
        for child in node.children:
            flattened_child = flatten_tree(child)
            if flattened_child.name == node.name:
                merged_children.extend(flattened_child.children)
            else:
                merged_children.append(flattened_child)
        return TreeNode(node.name, merged_children)
    else:
        node.children = [flatten_tree(child) for child in node.children]
        return node

def flatten_tree_one_level(node):
    if not node.children:
        return node

    if node.name in ("f_add", "f_mul"):

        merged_children = []
        for child in node.children:
            if child.name == node.name:

                merged_children.extend(child.children)
            else:

                merged_children.append(child)

        return TreeNode(node.name, merged_children)
    else:

        node.children = [flatten_tree_one_level(child) for child in node.children]
        return node

def break_equation(equation):
    sub_equation_list = [equation]
    equation = equation
    for child in equation.children: 
        sub_equation_list += break_equation(child) 
    return sub_equation_list

def unflatten(equation):

    if equation.name in {'f_add', 'f_mul'}:
        if len(equation.children) > 2:

            current_node = TreeNode(equation.name, [equation.children[0]])
            for child in equation.children[1:]:
                if isinstance(child, TreeNode) and child.name == equation.name:

                    current_node.children.extend(child.children)
                else:

                    current_node.children.append(child)

            for i in range(len(current_node.children)):
                current_node.children[i] = unflatten(current_node.children[i])
            return current_node
        else:

            return equation
    else:

        return equation

def reduce_brack_add(equation):
    number = []
    for i in range(len(equation.children)-1, -1, -1):
        if equation.children[i].name[:2] == "d_":
            number.append(int(equation.children.pop(i).name[2:]))

    number = tree_form("d_" + str(sum(number)))
    equation.children.append(number)
    return unflatten(equation)
    pass

def arrange_mul(equation):
    list_fx = []
    orig = copy.deepcopy(equation)
    for i in range(len(equation.children)-1,-1,-1):
        if "f_" in str_form(equation.children[i]).replace("f_add", "g_add").replace("f_pow", "g_pow").replace("f_mul", "g_mul"):
            list_fx.append(equation.children.pop(i))

    if len(list_fx) < 2 or len(equation.children) < 2:
        return unflatten(orig)

    fx_eq = TreeNode("f_mul", list_fx)

    return TreeNode("f_mul", [fx_eq, equation])
def collect(equation):
    output = []
    def collect_helper(equation, prev):
        if equation.name == "f_add" and prev == "f_pow":
            output.append(str_form(equation))
            return
        for child in equation.children:
            if child.name in ["f_add", "f_mul", "f_pow"]:
                collect_helper(child, equation.name)
            else:
                output.append(str_form(child))
    collect_helper(equation, equation.name)
    output = list(set(output))
    output = [x for x in output if tree_form(x).name[:2] != "d_"]
    return output

def operate_poly_mul(equation):

    length = len(equation.children[0].children)
    solution = []
    solution.append(TreeNode("f_mul", [equation.children[0].children[0], equation.children[1].children[0]]))
    for i in range(1,length):
        solution.append(TreeNode("f_add", [equation.children[0].children[i], equation.children[1].children[i]]))
    solution = TreeNode("f_poly", solution)
    return copy.deepcopy(solution)

def operate_poly_add(equation):
    length = len(equation.children[0].children)
    if all(str_form(equation.children[0].children[i]) == str_form(equation.children[1].children[i]) for i in range(1,length)):

        tmp = copy.deepcopy(equation)
        tmp.children[0].children[0] = TreeNode("f_add", [tmp.children[0].children[0], tmp.children[1].children[0]])
        return tmp.children[0]
    return None
def poly_add(equation):

    length = len(equation.children[0].children)
    dic_type = {}
    for i in range(len(equation.children)):
        tmp = copy.deepcopy(equation.children[i])
        tmp.children.pop(0)
        if str_form(tmp) in dic_type.keys():
            dic_type[str_form(tmp)] = TreeNode("f_add", [equation.children[i].children[0], dic_type[str_form(tmp)]])
        else:
            dic_type[str_form(tmp)] = equation.children[i].children[0]
    brac = []
    for key in dic_type.keys():
        tmp = tree_form(key)
        tmp.children = [dic_type[key]] + tmp.children
        brac.append(tmp)
    if len(brac) != 1:
        return TreeNode("f_add", brac)
    return brac[0]
def operate_poly_pow(equation):
    length = len(equation.children[0].children)
    if all(str_form(equation.children[1].children[i]) == "d_0" for i in range(1,length)):
        tmp = copy.deepcopy(equation)
        tmp.children[0].children[0] = TreeNode("f_pow", [equation.children[0].children[0], equation.children[1].children[0]])
        for i in range(1, length):
            tmp.children[0].children[i] = TreeNode("f_mul", [equation.children[0].children[i], equation.children[1].children[0]])
        return tmp.children[0]
    return None
def arithmetic(equation):
    tmp = sorted(search(str_form(equation), 1000, [None, None, None]), key=lambda x: len(string_equation(x)))
    if tmp != []:
        return tree_form(tmp[0])
    return equation

def change(equation):
    def change_helper(equation):
        output = None
        if len(equation.children) > 1 and equation.children[0].name == "f_poly" and equation.children[1].name == "f_poly":
            if equation.name == "f_mul":
                output = operate_poly_mul(equation)
            elif equation.name == "f_add":
                output = operate_poly_add(equation)
            elif equation.name == "f_pow":
                output = operate_poly_pow(equation)
        if output:
            return output
        return equation
    for item in [equation]+break_equation(equation):
        equation = replace(equation, item, change_helper(item))

    return equation

def reduce(eq):
    tmp2 = sorted([y for y in search(eq, 1000, ["formula-list-6/convert_division.txt", None, None]) if "f_div" not in y], key=lambda x: len(string_equation(x)))
    if tmp2 != []:
        eq = tmp2[0]
    term = collect(tree_form(eq))
    zero = []
    for i in range(len(term)+1):
        zero.append(tree_form("d_0"))
    eq = tree_form(eq.replace("d_", "s_"))

    zero[0] = tree_form("d_1")
    def find_sub(term):
        for i in range(len(term)):
            for j in range(len(term)):
                if i >= j:
                    continue
                if term[i] in [str_form(x) for x in break_equation(tree_form(term[j]))]:
                    term[i], term[j] = term[j], term[i]
                    return term
        return None
    while True:
        tmp4 = find_sub(term)
        if tmp4 is None:
            break
        term = tmp4
    for i in range(len(term)):
        zero[i+1] = tree_form("d_1")
        eq = replace(eq, tree_form(term[i].replace("d_", "s_")), TreeNode("f_poly", copy.deepcopy(zero)))
        zero[i+1] = tree_form("d_0")

    zero.pop(0)
    for i in range(-100, 100):
        eq = replace(eq, tree_form("s_" + str(i)), TreeNode("f_poly", [tree_form("d_" + str(i))] + copy.deepcopy(zero)))

    def calc_mul(eq):
        while True:
            orig = str_form(eq)
            eq = change(eq)
            eq = arithmetic(eq)
            if orig == str_form(eq):
                break
        return eq
    def calc_add(eq):
        while True:
            orig = str_form(eq)
            for q in break_equation(eq):
                if q.name == "f_add":
                    eq = replace(copy.deepcopy(eq), q, unflatten(poly_add(flatten_tree(q))))
            eq = arithmetic(eq)
            if orig == str_form(eq):
                break
        return eq
    eq = calc_mul(eq)
    eq = calc_add(eq)
    eq = calc_mul(eq)

    for i in range(-100, 100):
        eq = replace(eq, TreeNode("f_poly", [tree_form("d_" + str(i))] + copy.deepcopy(zero)),  tree_form("s_" + str(i)))

    for equation in copy.deepcopy(break_equation(eq)):
        if equation.name == "f_poly":
            brac = []
            if equation.children[0].name != "d_1":
                brac.append(equation.children[0])
            for i in range(1,len(equation.children)):
                if equation.children[i].name == "d_0":
                    continue
                elif equation.children[i].name == "d_1":
                    brac.append(tree_form(term[i-1]))
                else:
                    brac.append(TreeNode("f_pow", [tree_form(term[i-1]), copy.deepcopy(equation.children[i])]))
            if len(brac) == 1:
                brac = brac[0]
            elif len(brac) != 2:
                brac = unflatten(TreeNode("f_mul", brac))
            else:
                brac = TreeNode("f_mul", brac)
            eq = replace(eq, equation, brac)

    eq = str_form(eq).replace("s_", "d_")
    tmp2 = sorted(search(eq, 1000, ["formula-list-6/convert_division_2.txt", None, None]), key=lambda x: len(string_equation(x)))
    if tmp2 != []:
        eq = tmp2[0]
    return eq

def reduce_function(eq):

    eq = reduce(eq)

    eq = tree_form(eq)

    for item in collect(eq):
        item = tree_form(item)
        if item.name[:2] == "v_":
            continue
        eq = copy.deepcopy(replace(eq, item.children[0], tree_form(reduce(str_form(item.children[0])))))

    eq = str_form(eq)
    return eq

def build_tree(node):
    if isinstance(node, ast.BinOp):
        operator = node.op
        operator_name = {
            ast.Add: "f_add",
            ast.Sub: "f_sub",
            ast.Mult: "f_mul",
            ast.Div: "f_div",
            ast.Pow: "f_pow",
        }.get(type(operator), "f_unknown")
        tree_node = TreeNode(operator_name)
        tree_node.children.append(build_tree(node.left))
        tree_node.children.append(build_tree(node.right))
        return tree_node

    elif isinstance(node, ast.Num):  
        return TreeNode(f"d_{node.n}")

    elif isinstance(node, ast.Name):  
        return TreeNode(f"d_{node.id}")

    elif isinstance(node, ast.Call):  
        func_name = node.func.id
        tree_node = TreeNode(f"f_{func_name}")
        for arg in node.args:
            tree_node.children.append(build_tree(arg))
        return tree_node

    return TreeNode("d_unknown")

def parse_expression(expression):
    tree = ast.parse(expression, mode='eval')
    return build_tree(tree.body)

def parse_expression_main(eq):
    return str_form(parse_expression(eq.replace("^", "**"))).replace("d_x", "v_0")
def add_equation_to_log(equation):

    if 'equations' not in st.session_state:
        st.session_state.equations = []  

    st.session_state.equations.append(equation)
st.session_state.clear()

def communicate_with_math_ai(eq):
    try:
        eq = str_form(unflatten(tree_form(eq)))
        eq = str_form(TreeNode("f_dif", [tree_form(parse_expression_main(eq))]))
    except:
        return "parsing error"
    add_equation_to_log(string_equation(eq))
    try:
        eq = main_search(eq, "diff_formula.txt")
        eq = post_search(eq, "algebraic_formula.txt")
    except:
        return "equation not supported"
    try:
        eq = reduce_function(eq)
    except:
        return string_equation(eq)
    return string_equation(eq)

if 'final_result' not in st.session_state:
    st.session_state.final_result = None

if 'equations' not in st.session_state:
    st.session_state.equations = []

st.title("automatic differentiation wrt x")

equation_input = None
col1, col2 = st.columns([1, 5])  

with col1:
    st.latex(r'\frac{d}{dx}')

with col2:
    equation_input = st.text_input("enter your equation:", value="sin(cos(x))")

if equation_input:

    result = communicate_with_math_ai(equation_input)

    st.session_state.final_result = result
    st.success(f"valid equation: {equation_input}")

st.subheader("thought process")
if st.session_state.equations:

    for eq in st.session_state.equations:
        st.text(eq)  
else:
    st.text("no equations logged yet.")

st.subheader("final result")
if st.session_state.final_result is not None:

    st.text(f"{st.session_state.final_result}")

# differentiation Ai
# solve differentiation questions by rule based Ai
# the code is 800 lines of code. it leverages the data structure trees in order to do what it is able to do. there is a lot of underlying logic i have been developing for some months that is being used in this software. the most complicated part was the simplification of mathematical equations based like or unlike terms.
