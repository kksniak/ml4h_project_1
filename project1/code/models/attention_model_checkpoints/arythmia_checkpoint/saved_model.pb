ї‘
ЄО
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
Ы
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

R
Einsum
inputs"T*N
output"T"
equationstring"
Nint(0"	
Ttype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
М
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
Ѕ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И®
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.8.02v2.8.0-0-g3f878cff5b68№≥
z
conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d/kernel
s
!conv1d/kernel/Read/ReadVariableOpReadVariableOpconv1d/kernel*"
_output_shapes
:@*
dtype0
n
conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d/bias
g
conv1d/bias/Read/ReadVariableOpReadVariableOpconv1d/bias*
_output_shapes
:@*
dtype0
~
conv1d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv1d_1/kernel
w
#conv1d_1/kernel/Read/ReadVariableOpReadVariableOpconv1d_1/kernel*"
_output_shapes
:@*
dtype0
r
conv1d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_1/bias
k
!conv1d_1/bias/Read/ReadVariableOpReadVariableOpconv1d_1/bias*
_output_shapes
:*
dtype0
~
conv1d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv1d_2/kernel
w
#conv1d_2/kernel/Read/ReadVariableOpReadVariableOpconv1d_2/kernel*"
_output_shapes
:@*
dtype0
r
conv1d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d_2/bias
k
!conv1d_2/bias/Read/ReadVariableOpReadVariableOpconv1d_2/bias*
_output_shapes
:@*
dtype0
~
conv1d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv1d_3/kernel
w
#conv1d_3/kernel/Read/ReadVariableOpReadVariableOpconv1d_3/kernel*"
_output_shapes
:@*
dtype0
r
conv1d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_3/bias
k
!conv1d_3/bias/Read/ReadVariableOpReadVariableOpconv1d_3/bias*
_output_shapes
:*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ї@*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	ї@*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:@*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:@@*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:@*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:@*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
Ґ
!multi_head_attention/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!multi_head_attention/query/kernel
Ы
5multi_head_attention/query/kernel/Read/ReadVariableOpReadVariableOp!multi_head_attention/query/kernel*"
_output_shapes
:*
dtype0
Ъ
multi_head_attention/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*0
shared_name!multi_head_attention/query/bias
У
3multi_head_attention/query/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention/query/bias*
_output_shapes

:*
dtype0
Ю
multi_head_attention/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!multi_head_attention/key/kernel
Ч
3multi_head_attention/key/kernel/Read/ReadVariableOpReadVariableOpmulti_head_attention/key/kernel*"
_output_shapes
:*
dtype0
Ц
multi_head_attention/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_namemulti_head_attention/key/bias
П
1multi_head_attention/key/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention/key/bias*
_output_shapes

:*
dtype0
Ґ
!multi_head_attention/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!multi_head_attention/value/kernel
Ы
5multi_head_attention/value/kernel/Read/ReadVariableOpReadVariableOp!multi_head_attention/value/kernel*"
_output_shapes
:*
dtype0
Ъ
multi_head_attention/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*0
shared_name!multi_head_attention/value/bias
У
3multi_head_attention/value/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention/value/bias*
_output_shapes

:*
dtype0
Є
,multi_head_attention/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,multi_head_attention/attention_output/kernel
±
@multi_head_attention/attention_output/kernel/Read/ReadVariableOpReadVariableOp,multi_head_attention/attention_output/kernel*"
_output_shapes
:*
dtype0
ђ
*multi_head_attention/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*multi_head_attention/attention_output/bias
•
>multi_head_attention/attention_output/bias/Read/ReadVariableOpReadVariableOp*multi_head_attention/attention_output/bias*
_output_shapes
:*
dtype0
¶
#multi_head_attention_1/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#multi_head_attention_1/query/kernel
Я
7multi_head_attention_1/query/kernel/Read/ReadVariableOpReadVariableOp#multi_head_attention_1/query/kernel*"
_output_shapes
:*
dtype0
Ю
!multi_head_attention_1/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*2
shared_name#!multi_head_attention_1/query/bias
Ч
5multi_head_attention_1/query/bias/Read/ReadVariableOpReadVariableOp!multi_head_attention_1/query/bias*
_output_shapes

:*
dtype0
Ґ
!multi_head_attention_1/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!multi_head_attention_1/key/kernel
Ы
5multi_head_attention_1/key/kernel/Read/ReadVariableOpReadVariableOp!multi_head_attention_1/key/kernel*"
_output_shapes
:*
dtype0
Ъ
multi_head_attention_1/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*0
shared_name!multi_head_attention_1/key/bias
У
3multi_head_attention_1/key/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention_1/key/bias*
_output_shapes

:*
dtype0
¶
#multi_head_attention_1/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#multi_head_attention_1/value/kernel
Я
7multi_head_attention_1/value/kernel/Read/ReadVariableOpReadVariableOp#multi_head_attention_1/value/kernel*"
_output_shapes
:*
dtype0
Ю
!multi_head_attention_1/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*2
shared_name#!multi_head_attention_1/value/bias
Ч
5multi_head_attention_1/value/bias/Read/ReadVariableOpReadVariableOp!multi_head_attention_1/value/bias*
_output_shapes

:*
dtype0
Љ
.multi_head_attention_1/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.multi_head_attention_1/attention_output/kernel
µ
Bmulti_head_attention_1/attention_output/kernel/Read/ReadVariableOpReadVariableOp.multi_head_attention_1/attention_output/kernel*"
_output_shapes
:*
dtype0
∞
,multi_head_attention_1/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,multi_head_attention_1/attention_output/bias
©
@multi_head_attention_1/attention_output/bias/Read/ReadVariableOpReadVariableOp,multi_head_attention_1/attention_output/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
И
Adam/conv1d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv1d/kernel/m
Б
(Adam/conv1d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d/kernel/m*"
_output_shapes
:@*
dtype0
|
Adam/conv1d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/conv1d/bias/m
u
&Adam/conv1d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d/bias/m*
_output_shapes
:@*
dtype0
М
Adam/conv1d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv1d_1/kernel/m
Е
*Adam/conv1d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/kernel/m*"
_output_shapes
:@*
dtype0
А
Adam/conv1d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv1d_1/bias/m
y
(Adam/conv1d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/bias/m*
_output_shapes
:*
dtype0
М
Adam/conv1d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv1d_2/kernel/m
Е
*Adam/conv1d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_2/kernel/m*"
_output_shapes
:@*
dtype0
А
Adam/conv1d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv1d_2/bias/m
y
(Adam/conv1d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_2/bias/m*
_output_shapes
:@*
dtype0
М
Adam/conv1d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv1d_3/kernel/m
Е
*Adam/conv1d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_3/kernel/m*"
_output_shapes
:@*
dtype0
А
Adam/conv1d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv1d_3/bias/m
y
(Adam/conv1d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_3/bias/m*
_output_shapes
:*
dtype0
Г
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ї@*$
shared_nameAdam/dense/kernel/m
|
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes
:	ї@*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:@*
dtype0
Ж
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

:@@*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:@*
dtype0
Ж
Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes

:@*
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:*
dtype0
∞
(Adam/multi_head_attention/query/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/multi_head_attention/query/kernel/m
©
<Adam/multi_head_attention/query/kernel/m/Read/ReadVariableOpReadVariableOp(Adam/multi_head_attention/query/kernel/m*"
_output_shapes
:*
dtype0
®
&Adam/multi_head_attention/query/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&Adam/multi_head_attention/query/bias/m
°
:Adam/multi_head_attention/query/bias/m/Read/ReadVariableOpReadVariableOp&Adam/multi_head_attention/query/bias/m*
_output_shapes

:*
dtype0
ђ
&Adam/multi_head_attention/key/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&Adam/multi_head_attention/key/kernel/m
•
:Adam/multi_head_attention/key/kernel/m/Read/ReadVariableOpReadVariableOp&Adam/multi_head_attention/key/kernel/m*"
_output_shapes
:*
dtype0
§
$Adam/multi_head_attention/key/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*5
shared_name&$Adam/multi_head_attention/key/bias/m
Э
8Adam/multi_head_attention/key/bias/m/Read/ReadVariableOpReadVariableOp$Adam/multi_head_attention/key/bias/m*
_output_shapes

:*
dtype0
∞
(Adam/multi_head_attention/value/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/multi_head_attention/value/kernel/m
©
<Adam/multi_head_attention/value/kernel/m/Read/ReadVariableOpReadVariableOp(Adam/multi_head_attention/value/kernel/m*"
_output_shapes
:*
dtype0
®
&Adam/multi_head_attention/value/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&Adam/multi_head_attention/value/bias/m
°
:Adam/multi_head_attention/value/bias/m/Read/ReadVariableOpReadVariableOp&Adam/multi_head_attention/value/bias/m*
_output_shapes

:*
dtype0
∆
3Adam/multi_head_attention/attention_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53Adam/multi_head_attention/attention_output/kernel/m
њ
GAdam/multi_head_attention/attention_output/kernel/m/Read/ReadVariableOpReadVariableOp3Adam/multi_head_attention/attention_output/kernel/m*"
_output_shapes
:*
dtype0
Ї
1Adam/multi_head_attention/attention_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31Adam/multi_head_attention/attention_output/bias/m
≥
EAdam/multi_head_attention/attention_output/bias/m/Read/ReadVariableOpReadVariableOp1Adam/multi_head_attention/attention_output/bias/m*
_output_shapes
:*
dtype0
і
*Adam/multi_head_attention_1/query/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/multi_head_attention_1/query/kernel/m
≠
>Adam/multi_head_attention_1/query/kernel/m/Read/ReadVariableOpReadVariableOp*Adam/multi_head_attention_1/query/kernel/m*"
_output_shapes
:*
dtype0
ђ
(Adam/multi_head_attention_1/query/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*9
shared_name*(Adam/multi_head_attention_1/query/bias/m
•
<Adam/multi_head_attention_1/query/bias/m/Read/ReadVariableOpReadVariableOp(Adam/multi_head_attention_1/query/bias/m*
_output_shapes

:*
dtype0
∞
(Adam/multi_head_attention_1/key/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/multi_head_attention_1/key/kernel/m
©
<Adam/multi_head_attention_1/key/kernel/m/Read/ReadVariableOpReadVariableOp(Adam/multi_head_attention_1/key/kernel/m*"
_output_shapes
:*
dtype0
®
&Adam/multi_head_attention_1/key/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&Adam/multi_head_attention_1/key/bias/m
°
:Adam/multi_head_attention_1/key/bias/m/Read/ReadVariableOpReadVariableOp&Adam/multi_head_attention_1/key/bias/m*
_output_shapes

:*
dtype0
і
*Adam/multi_head_attention_1/value/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/multi_head_attention_1/value/kernel/m
≠
>Adam/multi_head_attention_1/value/kernel/m/Read/ReadVariableOpReadVariableOp*Adam/multi_head_attention_1/value/kernel/m*"
_output_shapes
:*
dtype0
ђ
(Adam/multi_head_attention_1/value/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*9
shared_name*(Adam/multi_head_attention_1/value/bias/m
•
<Adam/multi_head_attention_1/value/bias/m/Read/ReadVariableOpReadVariableOp(Adam/multi_head_attention_1/value/bias/m*
_output_shapes

:*
dtype0
 
5Adam/multi_head_attention_1/attention_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*F
shared_name75Adam/multi_head_attention_1/attention_output/kernel/m
√
IAdam/multi_head_attention_1/attention_output/kernel/m/Read/ReadVariableOpReadVariableOp5Adam/multi_head_attention_1/attention_output/kernel/m*"
_output_shapes
:*
dtype0
Њ
3Adam/multi_head_attention_1/attention_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53Adam/multi_head_attention_1/attention_output/bias/m
Ј
GAdam/multi_head_attention_1/attention_output/bias/m/Read/ReadVariableOpReadVariableOp3Adam/multi_head_attention_1/attention_output/bias/m*
_output_shapes
:*
dtype0
И
Adam/conv1d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv1d/kernel/v
Б
(Adam/conv1d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d/kernel/v*"
_output_shapes
:@*
dtype0
|
Adam/conv1d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/conv1d/bias/v
u
&Adam/conv1d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d/bias/v*
_output_shapes
:@*
dtype0
М
Adam/conv1d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv1d_1/kernel/v
Е
*Adam/conv1d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/kernel/v*"
_output_shapes
:@*
dtype0
А
Adam/conv1d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv1d_1/bias/v
y
(Adam/conv1d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/bias/v*
_output_shapes
:*
dtype0
М
Adam/conv1d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv1d_2/kernel/v
Е
*Adam/conv1d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_2/kernel/v*"
_output_shapes
:@*
dtype0
А
Adam/conv1d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv1d_2/bias/v
y
(Adam/conv1d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_2/bias/v*
_output_shapes
:@*
dtype0
М
Adam/conv1d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv1d_3/kernel/v
Е
*Adam/conv1d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_3/kernel/v*"
_output_shapes
:@*
dtype0
А
Adam/conv1d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv1d_3/bias/v
y
(Adam/conv1d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_3/bias/v*
_output_shapes
:*
dtype0
Г
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ї@*$
shared_nameAdam/dense/kernel/v
|
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes
:	ї@*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:@*
dtype0
Ж
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

:@@*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:@*
dtype0
Ж
Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes

:@*
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:*
dtype0
∞
(Adam/multi_head_attention/query/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/multi_head_attention/query/kernel/v
©
<Adam/multi_head_attention/query/kernel/v/Read/ReadVariableOpReadVariableOp(Adam/multi_head_attention/query/kernel/v*"
_output_shapes
:*
dtype0
®
&Adam/multi_head_attention/query/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&Adam/multi_head_attention/query/bias/v
°
:Adam/multi_head_attention/query/bias/v/Read/ReadVariableOpReadVariableOp&Adam/multi_head_attention/query/bias/v*
_output_shapes

:*
dtype0
ђ
&Adam/multi_head_attention/key/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&Adam/multi_head_attention/key/kernel/v
•
:Adam/multi_head_attention/key/kernel/v/Read/ReadVariableOpReadVariableOp&Adam/multi_head_attention/key/kernel/v*"
_output_shapes
:*
dtype0
§
$Adam/multi_head_attention/key/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*5
shared_name&$Adam/multi_head_attention/key/bias/v
Э
8Adam/multi_head_attention/key/bias/v/Read/ReadVariableOpReadVariableOp$Adam/multi_head_attention/key/bias/v*
_output_shapes

:*
dtype0
∞
(Adam/multi_head_attention/value/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/multi_head_attention/value/kernel/v
©
<Adam/multi_head_attention/value/kernel/v/Read/ReadVariableOpReadVariableOp(Adam/multi_head_attention/value/kernel/v*"
_output_shapes
:*
dtype0
®
&Adam/multi_head_attention/value/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&Adam/multi_head_attention/value/bias/v
°
:Adam/multi_head_attention/value/bias/v/Read/ReadVariableOpReadVariableOp&Adam/multi_head_attention/value/bias/v*
_output_shapes

:*
dtype0
∆
3Adam/multi_head_attention/attention_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53Adam/multi_head_attention/attention_output/kernel/v
њ
GAdam/multi_head_attention/attention_output/kernel/v/Read/ReadVariableOpReadVariableOp3Adam/multi_head_attention/attention_output/kernel/v*"
_output_shapes
:*
dtype0
Ї
1Adam/multi_head_attention/attention_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31Adam/multi_head_attention/attention_output/bias/v
≥
EAdam/multi_head_attention/attention_output/bias/v/Read/ReadVariableOpReadVariableOp1Adam/multi_head_attention/attention_output/bias/v*
_output_shapes
:*
dtype0
і
*Adam/multi_head_attention_1/query/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/multi_head_attention_1/query/kernel/v
≠
>Adam/multi_head_attention_1/query/kernel/v/Read/ReadVariableOpReadVariableOp*Adam/multi_head_attention_1/query/kernel/v*"
_output_shapes
:*
dtype0
ђ
(Adam/multi_head_attention_1/query/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*9
shared_name*(Adam/multi_head_attention_1/query/bias/v
•
<Adam/multi_head_attention_1/query/bias/v/Read/ReadVariableOpReadVariableOp(Adam/multi_head_attention_1/query/bias/v*
_output_shapes

:*
dtype0
∞
(Adam/multi_head_attention_1/key/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/multi_head_attention_1/key/kernel/v
©
<Adam/multi_head_attention_1/key/kernel/v/Read/ReadVariableOpReadVariableOp(Adam/multi_head_attention_1/key/kernel/v*"
_output_shapes
:*
dtype0
®
&Adam/multi_head_attention_1/key/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&Adam/multi_head_attention_1/key/bias/v
°
:Adam/multi_head_attention_1/key/bias/v/Read/ReadVariableOpReadVariableOp&Adam/multi_head_attention_1/key/bias/v*
_output_shapes

:*
dtype0
і
*Adam/multi_head_attention_1/value/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/multi_head_attention_1/value/kernel/v
≠
>Adam/multi_head_attention_1/value/kernel/v/Read/ReadVariableOpReadVariableOp*Adam/multi_head_attention_1/value/kernel/v*"
_output_shapes
:*
dtype0
ђ
(Adam/multi_head_attention_1/value/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*9
shared_name*(Adam/multi_head_attention_1/value/bias/v
•
<Adam/multi_head_attention_1/value/bias/v/Read/ReadVariableOpReadVariableOp(Adam/multi_head_attention_1/value/bias/v*
_output_shapes

:*
dtype0
 
5Adam/multi_head_attention_1/attention_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*F
shared_name75Adam/multi_head_attention_1/attention_output/kernel/v
√
IAdam/multi_head_attention_1/attention_output/kernel/v/Read/ReadVariableOpReadVariableOp5Adam/multi_head_attention_1/attention_output/kernel/v*"
_output_shapes
:*
dtype0
Њ
3Adam/multi_head_attention_1/attention_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53Adam/multi_head_attention_1/attention_output/bias/v
Ј
GAdam/multi_head_attention_1/attention_output/bias/v/Read/ReadVariableOpReadVariableOp3Adam/multi_head_attention_1/attention_output/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
нё
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Іё
valueЬёBШё BРё
®
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer-10
layer_with_weights-5
layer-11
layer-12
layer-13
layer-14
layer_with_weights-6
layer-15
layer_with_weights-7
layer-16
layer_with_weights-8
layer-17
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
щ
_query_dense

_key_dense
_value_dense
_softmax
 _dropout_layer
!_output_dense
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses*

(	keras_api* 
¶

)kernel
*bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses*
•
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5_random_generator
6__call__
*7&call_and_return_all_conditional_losses* 
¶

8kernel
9bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses*

@	keras_api* 
щ
A_query_dense
B
_key_dense
C_value_dense
D_softmax
E_dropout_layer
F_output_dense
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses*

M	keras_api* 
¶

Nkernel
Obias
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses*
•
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z_random_generator
[__call__
*\&call_and_return_all_conditional_losses* 
¶

]kernel
^bias
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses*

e	keras_api* 
О
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses* 
•
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p_random_generator
q__call__
*r&call_and_return_all_conditional_losses* 
¶

skernel
tbias
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses*
©

{kernel
|bias
}	variables
~trainable_variables
regularization_losses
А	keras_api
Б__call__
+В&call_and_return_all_conditional_losses*
Ѓ
Гkernel
	Дbias
Е	variables
Жtrainable_variables
Зregularization_losses
И	keras_api
Й__call__
+К&call_and_return_all_conditional_losses*
≈
	Лiter
Мbeta_1
Нbeta_2

Оdecay
Пlearning_rate)mИ*mЙ8mК9mЛNmМOmН]mО^mПsmРtmС{mТ|mУ	ГmФ	ДmХ	РmЦ	СmЧ	ТmШ	УmЩ	ФmЪ	ХmЫ	ЦmЬ	ЧmЭ	ШmЮ	ЩmЯ	Ъm†	Ыm°	ЬmҐ	Эm£	Юm§	Яm•)v¶*vІ8v®9v©Nv™OvЂ]vђ^v≠svЃtvѓ{v∞|v±	Гv≤	Дv≥	Рvі	Сvµ	Тvґ	УvЈ	ФvЄ	Хvє	ЦvЇ	Чvї	ШvЉ	Щvљ	ЪvЊ	Ыvњ	Ьvј	ЭvЅ	Юv¬	Яv√*
ь
Р0
С1
Т2
У3
Ф4
Х5
Ц6
Ч7
)8
*9
810
911
Ш12
Щ13
Ъ14
Ы15
Ь16
Э17
Ю18
Я19
N20
O21
]22
^23
s24
t25
{26
|27
Г28
Д29*
ь
Р0
С1
Т2
У3
Ф4
Х5
Ц6
Ч7
)8
*9
810
911
Ш12
Щ13
Ъ14
Ы15
Ь16
Э17
Ю18
Я19
N20
O21
]22
^23
s24
t25
{26
|27
Г28
Д29*
* 
µ
†non_trainable_variables
°layers
Ґmetrics
 £layer_regularization_losses
§layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

•serving_default* 
б
¶partial_output_shape
Іfull_output_shape
Рkernel
	Сbias
®	variables
©trainable_variables
™regularization_losses
Ђ	keras_api
ђ__call__
+≠&call_and_return_all_conditional_losses*
б
Ѓpartial_output_shape
ѓfull_output_shape
Тkernel
	Уbias
∞	variables
±trainable_variables
≤regularization_losses
≥	keras_api
і__call__
+µ&call_and_return_all_conditional_losses*
б
ґpartial_output_shape
Јfull_output_shape
Фkernel
	Хbias
Є	variables
єtrainable_variables
Їregularization_losses
ї	keras_api
Љ__call__
+љ&call_and_return_all_conditional_losses*
Ф
Њ	variables
њtrainable_variables
јregularization_losses
Ѕ	keras_api
¬__call__
+√&call_and_return_all_conditional_losses* 
ђ
ƒ	variables
≈trainable_variables
∆regularization_losses
«	keras_api
»_random_generator
…__call__
+ &call_and_return_all_conditional_losses* 
б
Ћpartial_output_shape
ћfull_output_shape
Цkernel
	Чbias
Ќ	variables
ќtrainable_variables
ѕregularization_losses
–	keras_api
—__call__
+“&call_and_return_all_conditional_losses*
D
Р0
С1
Т2
У3
Ф4
Х5
Ц6
Ч7*
D
Р0
С1
Т2
У3
Ф4
Х5
Ц6
Ч7*
* 
Ш
”non_trainable_variables
‘layers
’metrics
 ÷layer_regularization_losses
„layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses*
* 
* 
* 
]W
VARIABLE_VALUEconv1d/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv1d/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

)0
*1*

)0
*1*
* 
Ш
Ўnon_trainable_variables
ўlayers
Џmetrics
 џlayer_regularization_losses
№layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
Ц
Ёnon_trainable_variables
ёlayers
яmetrics
 аlayer_regularization_losses
бlayer_metrics
1	variables
2trainable_variables
3regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses* 
* 
* 
* 
_Y
VARIABLE_VALUEconv1d_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv1d_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

80
91*

80
91*
* 
Ш
вnon_trainable_variables
гlayers
дmetrics
 еlayer_regularization_losses
жlayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*
* 
* 
* 
б
зpartial_output_shape
иfull_output_shape
Шkernel
	Щbias
й	variables
кtrainable_variables
лregularization_losses
м	keras_api
н__call__
+о&call_and_return_all_conditional_losses*
б
пpartial_output_shape
рfull_output_shape
Ъkernel
	Ыbias
с	variables
тtrainable_variables
уregularization_losses
ф	keras_api
х__call__
+ц&call_and_return_all_conditional_losses*
б
чpartial_output_shape
шfull_output_shape
Ьkernel
	Эbias
щ	variables
ъtrainable_variables
ыregularization_losses
ь	keras_api
э__call__
+ю&call_and_return_all_conditional_losses*
Ф
€	variables
Аtrainable_variables
Бregularization_losses
В	keras_api
Г__call__
+Д&call_and_return_all_conditional_losses* 
ђ
Е	variables
Жtrainable_variables
Зregularization_losses
И	keras_api
Й_random_generator
К__call__
+Л&call_and_return_all_conditional_losses* 
б
Мpartial_output_shape
Нfull_output_shape
Юkernel
	Яbias
О	variables
Пtrainable_variables
Рregularization_losses
С	keras_api
Т__call__
+У&call_and_return_all_conditional_losses*
D
Ш0
Щ1
Ъ2
Ы3
Ь4
Э5
Ю6
Я7*
D
Ш0
Щ1
Ъ2
Ы3
Ь4
Э5
Ю6
Я7*
* 
Ш
Фnon_trainable_variables
Хlayers
Цmetrics
 Чlayer_regularization_losses
Шlayer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses*
* 
* 
* 
_Y
VARIABLE_VALUEconv1d_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv1d_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

N0
O1*

N0
O1*
* 
Ш
Щnon_trainable_variables
Ъlayers
Ыmetrics
 Ьlayer_regularization_losses
Эlayer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
Ц
Юnon_trainable_variables
Яlayers
†metrics
 °layer_regularization_losses
Ґlayer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses* 
* 
* 
* 
_Y
VARIABLE_VALUEconv1d_3/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv1d_3/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

]0
^1*

]0
^1*
* 
Ш
£non_trainable_variables
§layers
•metrics
 ¶layer_regularization_losses
Іlayer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
Ц
®non_trainable_variables
©layers
™metrics
 Ђlayer_regularization_losses
ђlayer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
Ц
≠non_trainable_variables
Ѓlayers
ѓmetrics
 ∞layer_regularization_losses
±layer_metrics
l	variables
mtrainable_variables
nregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses* 
* 
* 
* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

s0
t1*

s0
t1*
* 
Ш
≤non_trainable_variables
≥layers
іmetrics
 µlayer_regularization_losses
ґlayer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses*
* 
* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

{0
|1*

{0
|1*
* 
Ы
Јnon_trainable_variables
Єlayers
єmetrics
 Їlayer_regularization_losses
їlayer_metrics
}	variables
~trainable_variables
regularization_losses
Б__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses*
* 
* 
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*

Г0
Д1*

Г0
Д1*
* 
Ю
Љnon_trainable_variables
љlayers
Њmetrics
 њlayer_regularization_losses
јlayer_metrics
Е	variables
Жtrainable_variables
Зregularization_losses
Й__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!multi_head_attention/query/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEmulti_head_attention/query/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEmulti_head_attention/key/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEmulti_head_attention/key/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!multi_head_attention/value/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEmulti_head_attention/value/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE,multi_head_attention/attention_output/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUE*multi_head_attention/attention_output/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#multi_head_attention_1/query/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!multi_head_attention_1/query/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!multi_head_attention_1/key/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEmulti_head_attention_1/key/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#multi_head_attention_1/value/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!multi_head_attention_1/value/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE.multi_head_attention_1/attention_output/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE,multi_head_attention_1/attention_output/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
* 
К
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17*

Ѕ0
¬1*
* 
* 
* 
* 
* 

Р0
С1*

Р0
С1*
* 
Ю
√non_trainable_variables
ƒlayers
≈metrics
 ∆layer_regularization_losses
«layer_metrics
®	variables
©trainable_variables
™regularization_losses
ђ__call__
+≠&call_and_return_all_conditional_losses
'≠"call_and_return_conditional_losses*
* 
* 
* 
* 

Т0
У1*

Т0
У1*
* 
Ю
»non_trainable_variables
…layers
 metrics
 Ћlayer_regularization_losses
ћlayer_metrics
∞	variables
±trainable_variables
≤regularization_losses
і__call__
+µ&call_and_return_all_conditional_losses
'µ"call_and_return_conditional_losses*
* 
* 
* 
* 

Ф0
Х1*

Ф0
Х1*
* 
Ю
Ќnon_trainable_variables
ќlayers
ѕmetrics
 –layer_regularization_losses
—layer_metrics
Є	variables
єtrainable_variables
Їregularization_losses
Љ__call__
+љ&call_and_return_all_conditional_losses
'љ"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
Ь
“non_trainable_variables
”layers
‘metrics
 ’layer_regularization_losses
÷layer_metrics
Њ	variables
њtrainable_variables
јregularization_losses
¬__call__
+√&call_and_return_all_conditional_losses
'√"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
Ь
„non_trainable_variables
Ўlayers
ўmetrics
 Џlayer_regularization_losses
џlayer_metrics
ƒ	variables
≈trainable_variables
∆regularization_losses
…__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

Ц0
Ч1*

Ц0
Ч1*
* 
Ю
№non_trainable_variables
Ёlayers
ёmetrics
 яlayer_regularization_losses
аlayer_metrics
Ќ	variables
ќtrainable_variables
ѕregularization_losses
—__call__
+“&call_and_return_all_conditional_losses
'“"call_and_return_conditional_losses*
* 
* 
* 
.
0
1
2
3
 4
!5*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

Ш0
Щ1*

Ш0
Щ1*
* 
Ю
бnon_trainable_variables
вlayers
гmetrics
 дlayer_regularization_losses
еlayer_metrics
й	variables
кtrainable_variables
лregularization_losses
н__call__
+о&call_and_return_all_conditional_losses
'о"call_and_return_conditional_losses*
* 
* 
* 
* 

Ъ0
Ы1*

Ъ0
Ы1*
* 
Ю
жnon_trainable_variables
зlayers
иmetrics
 йlayer_regularization_losses
кlayer_metrics
с	variables
тtrainable_variables
уregularization_losses
х__call__
+ц&call_and_return_all_conditional_losses
'ц"call_and_return_conditional_losses*
* 
* 
* 
* 

Ь0
Э1*

Ь0
Э1*
* 
Ю
лnon_trainable_variables
мlayers
нmetrics
 оlayer_regularization_losses
пlayer_metrics
щ	variables
ъtrainable_variables
ыregularization_losses
э__call__
+ю&call_and_return_all_conditional_losses
'ю"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
Ь
рnon_trainable_variables
сlayers
тmetrics
 уlayer_regularization_losses
фlayer_metrics
€	variables
Аtrainable_variables
Бregularization_losses
Г__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
Ь
хnon_trainable_variables
цlayers
чmetrics
 шlayer_regularization_losses
щlayer_metrics
Е	variables
Жtrainable_variables
Зregularization_losses
К__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

Ю0
Я1*

Ю0
Я1*
* 
Ю
ъnon_trainable_variables
ыlayers
ьmetrics
 эlayer_regularization_losses
юlayer_metrics
О	variables
Пtrainable_variables
Рregularization_losses
Т__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses*
* 
* 
* 
.
A0
B1
C2
D3
E4
F5*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<

€total

Аcount
Б	variables
В	keras_api*
M

Гtotal

Дcount
Е
_fn_kwargs
Ж	variables
З	keras_api*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

€0
А1*

Б	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

Г0
Д1*

Ж	variables*
Аz
VARIABLE_VALUEAdam/conv1d/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/conv1d/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/conv1d_1/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv1d_1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/conv1d_2/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv1d_2/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/conv1d_3/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv1d_3/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUE(Adam/multi_head_attention/query/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUE&Adam/multi_head_attention/query/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUE&Adam/multi_head_attention/key/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUE$Adam/multi_head_attention/key/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUE(Adam/multi_head_attention/value/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUE&Adam/multi_head_attention/value/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
РЙ
VARIABLE_VALUE3Adam/multi_head_attention/attention_output/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ОЗ
VARIABLE_VALUE1Adam/multi_head_attention/attention_output/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ИБ
VARIABLE_VALUE*Adam/multi_head_attention_1/query/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Е
VARIABLE_VALUE(Adam/multi_head_attention_1/query/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Е
VARIABLE_VALUE(Adam/multi_head_attention_1/key/kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUE&Adam/multi_head_attention_1/key/bias/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ИБ
VARIABLE_VALUE*Adam/multi_head_attention_1/value/kernel/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Е
VARIABLE_VALUE(Adam/multi_head_attention_1/value/bias/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
УМ
VARIABLE_VALUE5Adam/multi_head_attention_1/attention_output/kernel/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
СК
VARIABLE_VALUE3Adam/multi_head_attention_1/attention_output/bias/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/conv1d/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/conv1d/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/conv1d_1/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv1d_1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/conv1d_2/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv1d_2/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/conv1d_3/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv1d_3/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUE(Adam/multi_head_attention/query/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUE&Adam/multi_head_attention/query/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUE&Adam/multi_head_attention/key/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUE$Adam/multi_head_attention/key/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUE(Adam/multi_head_attention/value/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUE&Adam/multi_head_attention/value/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
РЙ
VARIABLE_VALUE3Adam/multi_head_attention/attention_output/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ОЗ
VARIABLE_VALUE1Adam/multi_head_attention/attention_output/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ИБ
VARIABLE_VALUE*Adam/multi_head_attention_1/query/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Е
VARIABLE_VALUE(Adam/multi_head_attention_1/query/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Е
VARIABLE_VALUE(Adam/multi_head_attention_1/key/kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUE&Adam/multi_head_attention_1/key/bias/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ИБ
VARIABLE_VALUE*Adam/multi_head_attention_1/value/kernel/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Е
VARIABLE_VALUE(Adam/multi_head_attention_1/value/bias/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
УМ
VARIABLE_VALUE5Adam/multi_head_attention_1/attention_output/kernel/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
СК
VARIABLE_VALUE3Adam/multi_head_attention_1/attention_output/bias/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Д
serving_default_input_1Placeholder*,
_output_shapes
:€€€€€€€€€ї*
dtype0*!
shape:€€€€€€€€€ї
Ч	
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1!multi_head_attention/query/kernelmulti_head_attention/query/biasmulti_head_attention/key/kernelmulti_head_attention/key/bias!multi_head_attention/value/kernelmulti_head_attention/value/bias,multi_head_attention/attention_output/kernel*multi_head_attention/attention_output/biasconv1d/kernelconv1d/biasconv1d_1/kernelconv1d_1/bias#multi_head_attention_1/query/kernel!multi_head_attention_1/query/bias!multi_head_attention_1/key/kernelmulti_head_attention_1/key/bias#multi_head_attention_1/value/kernel!multi_head_attention_1/value/bias.multi_head_attention_1/attention_output/kernel,multi_head_attention_1/attention_output/biasconv1d_2/kernelconv1d_2/biasconv1d_3/kernelconv1d_3/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*@
_read_only_resource_inputs"
 	
*0
config_proto 

CPU

GPU2*0J 8В *-
f(R&
$__inference_signature_wrapper_189769
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Є)
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv1d/kernel/Read/ReadVariableOpconv1d/bias/Read/ReadVariableOp#conv1d_1/kernel/Read/ReadVariableOp!conv1d_1/bias/Read/ReadVariableOp#conv1d_2/kernel/Read/ReadVariableOp!conv1d_2/bias/Read/ReadVariableOp#conv1d_3/kernel/Read/ReadVariableOp!conv1d_3/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp5multi_head_attention/query/kernel/Read/ReadVariableOp3multi_head_attention/query/bias/Read/ReadVariableOp3multi_head_attention/key/kernel/Read/ReadVariableOp1multi_head_attention/key/bias/Read/ReadVariableOp5multi_head_attention/value/kernel/Read/ReadVariableOp3multi_head_attention/value/bias/Read/ReadVariableOp@multi_head_attention/attention_output/kernel/Read/ReadVariableOp>multi_head_attention/attention_output/bias/Read/ReadVariableOp7multi_head_attention_1/query/kernel/Read/ReadVariableOp5multi_head_attention_1/query/bias/Read/ReadVariableOp5multi_head_attention_1/key/kernel/Read/ReadVariableOp3multi_head_attention_1/key/bias/Read/ReadVariableOp7multi_head_attention_1/value/kernel/Read/ReadVariableOp5multi_head_attention_1/value/bias/Read/ReadVariableOpBmulti_head_attention_1/attention_output/kernel/Read/ReadVariableOp@multi_head_attention_1/attention_output/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp(Adam/conv1d/kernel/m/Read/ReadVariableOp&Adam/conv1d/bias/m/Read/ReadVariableOp*Adam/conv1d_1/kernel/m/Read/ReadVariableOp(Adam/conv1d_1/bias/m/Read/ReadVariableOp*Adam/conv1d_2/kernel/m/Read/ReadVariableOp(Adam/conv1d_2/bias/m/Read/ReadVariableOp*Adam/conv1d_3/kernel/m/Read/ReadVariableOp(Adam/conv1d_3/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp<Adam/multi_head_attention/query/kernel/m/Read/ReadVariableOp:Adam/multi_head_attention/query/bias/m/Read/ReadVariableOp:Adam/multi_head_attention/key/kernel/m/Read/ReadVariableOp8Adam/multi_head_attention/key/bias/m/Read/ReadVariableOp<Adam/multi_head_attention/value/kernel/m/Read/ReadVariableOp:Adam/multi_head_attention/value/bias/m/Read/ReadVariableOpGAdam/multi_head_attention/attention_output/kernel/m/Read/ReadVariableOpEAdam/multi_head_attention/attention_output/bias/m/Read/ReadVariableOp>Adam/multi_head_attention_1/query/kernel/m/Read/ReadVariableOp<Adam/multi_head_attention_1/query/bias/m/Read/ReadVariableOp<Adam/multi_head_attention_1/key/kernel/m/Read/ReadVariableOp:Adam/multi_head_attention_1/key/bias/m/Read/ReadVariableOp>Adam/multi_head_attention_1/value/kernel/m/Read/ReadVariableOp<Adam/multi_head_attention_1/value/bias/m/Read/ReadVariableOpIAdam/multi_head_attention_1/attention_output/kernel/m/Read/ReadVariableOpGAdam/multi_head_attention_1/attention_output/bias/m/Read/ReadVariableOp(Adam/conv1d/kernel/v/Read/ReadVariableOp&Adam/conv1d/bias/v/Read/ReadVariableOp*Adam/conv1d_1/kernel/v/Read/ReadVariableOp(Adam/conv1d_1/bias/v/Read/ReadVariableOp*Adam/conv1d_2/kernel/v/Read/ReadVariableOp(Adam/conv1d_2/bias/v/Read/ReadVariableOp*Adam/conv1d_3/kernel/v/Read/ReadVariableOp(Adam/conv1d_3/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp<Adam/multi_head_attention/query/kernel/v/Read/ReadVariableOp:Adam/multi_head_attention/query/bias/v/Read/ReadVariableOp:Adam/multi_head_attention/key/kernel/v/Read/ReadVariableOp8Adam/multi_head_attention/key/bias/v/Read/ReadVariableOp<Adam/multi_head_attention/value/kernel/v/Read/ReadVariableOp:Adam/multi_head_attention/value/bias/v/Read/ReadVariableOpGAdam/multi_head_attention/attention_output/kernel/v/Read/ReadVariableOpEAdam/multi_head_attention/attention_output/bias/v/Read/ReadVariableOp>Adam/multi_head_attention_1/query/kernel/v/Read/ReadVariableOp<Adam/multi_head_attention_1/query/bias/v/Read/ReadVariableOp<Adam/multi_head_attention_1/key/kernel/v/Read/ReadVariableOp:Adam/multi_head_attention_1/key/bias/v/Read/ReadVariableOp>Adam/multi_head_attention_1/value/kernel/v/Read/ReadVariableOp<Adam/multi_head_attention_1/value/bias/v/Read/ReadVariableOpIAdam/multi_head_attention_1/attention_output/kernel/v/Read/ReadVariableOpGAdam/multi_head_attention_1/attention_output/bias/v/Read/ReadVariableOpConst*p
Tini
g2e	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference__traced_save_190583
ч
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d/kernelconv1d/biasconv1d_1/kernelconv1d_1/biasconv1d_2/kernelconv1d_2/biasconv1d_3/kernelconv1d_3/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate!multi_head_attention/query/kernelmulti_head_attention/query/biasmulti_head_attention/key/kernelmulti_head_attention/key/bias!multi_head_attention/value/kernelmulti_head_attention/value/bias,multi_head_attention/attention_output/kernel*multi_head_attention/attention_output/bias#multi_head_attention_1/query/kernel!multi_head_attention_1/query/bias!multi_head_attention_1/key/kernelmulti_head_attention_1/key/bias#multi_head_attention_1/value/kernel!multi_head_attention_1/value/bias.multi_head_attention_1/attention_output/kernel,multi_head_attention_1/attention_output/biastotalcounttotal_1count_1Adam/conv1d/kernel/mAdam/conv1d/bias/mAdam/conv1d_1/kernel/mAdam/conv1d_1/bias/mAdam/conv1d_2/kernel/mAdam/conv1d_2/bias/mAdam/conv1d_3/kernel/mAdam/conv1d_3/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/dense_2/kernel/mAdam/dense_2/bias/m(Adam/multi_head_attention/query/kernel/m&Adam/multi_head_attention/query/bias/m&Adam/multi_head_attention/key/kernel/m$Adam/multi_head_attention/key/bias/m(Adam/multi_head_attention/value/kernel/m&Adam/multi_head_attention/value/bias/m3Adam/multi_head_attention/attention_output/kernel/m1Adam/multi_head_attention/attention_output/bias/m*Adam/multi_head_attention_1/query/kernel/m(Adam/multi_head_attention_1/query/bias/m(Adam/multi_head_attention_1/key/kernel/m&Adam/multi_head_attention_1/key/bias/m*Adam/multi_head_attention_1/value/kernel/m(Adam/multi_head_attention_1/value/bias/m5Adam/multi_head_attention_1/attention_output/kernel/m3Adam/multi_head_attention_1/attention_output/bias/mAdam/conv1d/kernel/vAdam/conv1d/bias/vAdam/conv1d_1/kernel/vAdam/conv1d_1/bias/vAdam/conv1d_2/kernel/vAdam/conv1d_2/bias/vAdam/conv1d_3/kernel/vAdam/conv1d_3/bias/vAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/dense_2/kernel/vAdam/dense_2/bias/v(Adam/multi_head_attention/query/kernel/v&Adam/multi_head_attention/query/bias/v&Adam/multi_head_attention/key/kernel/v$Adam/multi_head_attention/key/bias/v(Adam/multi_head_attention/value/kernel/v&Adam/multi_head_attention/value/bias/v3Adam/multi_head_attention/attention_output/kernel/v1Adam/multi_head_attention/attention_output/bias/v*Adam/multi_head_attention_1/query/kernel/v(Adam/multi_head_attention_1/query/bias/v(Adam/multi_head_attention_1/key/kernel/v&Adam/multi_head_attention_1/key/bias/v*Adam/multi_head_attention_1/value/kernel/v(Adam/multi_head_attention_1/value/bias/v5Adam/multi_head_attention_1/attention_output/kernel/v3Adam/multi_head_attention_1/attention_output/bias/v*o
Tinh
f2d*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *+
f&R$
"__inference__traced_restore_190890Ю§
ДТ
јE
"__inference__traced_restore_190890
file_prefix4
assignvariableop_conv1d_kernel:@,
assignvariableop_1_conv1d_bias:@8
"assignvariableop_2_conv1d_1_kernel:@.
 assignvariableop_3_conv1d_1_bias:8
"assignvariableop_4_conv1d_2_kernel:@.
 assignvariableop_5_conv1d_2_bias:@8
"assignvariableop_6_conv1d_3_kernel:@.
 assignvariableop_7_conv1d_3_bias:2
assignvariableop_8_dense_kernel:	ї@+
assignvariableop_9_dense_bias:@4
"assignvariableop_10_dense_1_kernel:@@.
 assignvariableop_11_dense_1_bias:@4
"assignvariableop_12_dense_2_kernel:@.
 assignvariableop_13_dense_2_bias:'
assignvariableop_14_adam_iter:	 )
assignvariableop_15_adam_beta_1: )
assignvariableop_16_adam_beta_2: (
assignvariableop_17_adam_decay: 0
&assignvariableop_18_adam_learning_rate: K
5assignvariableop_19_multi_head_attention_query_kernel:E
3assignvariableop_20_multi_head_attention_query_bias:I
3assignvariableop_21_multi_head_attention_key_kernel:C
1assignvariableop_22_multi_head_attention_key_bias:K
5assignvariableop_23_multi_head_attention_value_kernel:E
3assignvariableop_24_multi_head_attention_value_bias:V
@assignvariableop_25_multi_head_attention_attention_output_kernel:L
>assignvariableop_26_multi_head_attention_attention_output_bias:M
7assignvariableop_27_multi_head_attention_1_query_kernel:G
5assignvariableop_28_multi_head_attention_1_query_bias:K
5assignvariableop_29_multi_head_attention_1_key_kernel:E
3assignvariableop_30_multi_head_attention_1_key_bias:M
7assignvariableop_31_multi_head_attention_1_value_kernel:G
5assignvariableop_32_multi_head_attention_1_value_bias:X
Bassignvariableop_33_multi_head_attention_1_attention_output_kernel:N
@assignvariableop_34_multi_head_attention_1_attention_output_bias:#
assignvariableop_35_total: #
assignvariableop_36_count: %
assignvariableop_37_total_1: %
assignvariableop_38_count_1: >
(assignvariableop_39_adam_conv1d_kernel_m:@4
&assignvariableop_40_adam_conv1d_bias_m:@@
*assignvariableop_41_adam_conv1d_1_kernel_m:@6
(assignvariableop_42_adam_conv1d_1_bias_m:@
*assignvariableop_43_adam_conv1d_2_kernel_m:@6
(assignvariableop_44_adam_conv1d_2_bias_m:@@
*assignvariableop_45_adam_conv1d_3_kernel_m:@6
(assignvariableop_46_adam_conv1d_3_bias_m::
'assignvariableop_47_adam_dense_kernel_m:	ї@3
%assignvariableop_48_adam_dense_bias_m:@;
)assignvariableop_49_adam_dense_1_kernel_m:@@5
'assignvariableop_50_adam_dense_1_bias_m:@;
)assignvariableop_51_adam_dense_2_kernel_m:@5
'assignvariableop_52_adam_dense_2_bias_m:R
<assignvariableop_53_adam_multi_head_attention_query_kernel_m:L
:assignvariableop_54_adam_multi_head_attention_query_bias_m:P
:assignvariableop_55_adam_multi_head_attention_key_kernel_m:J
8assignvariableop_56_adam_multi_head_attention_key_bias_m:R
<assignvariableop_57_adam_multi_head_attention_value_kernel_m:L
:assignvariableop_58_adam_multi_head_attention_value_bias_m:]
Gassignvariableop_59_adam_multi_head_attention_attention_output_kernel_m:S
Eassignvariableop_60_adam_multi_head_attention_attention_output_bias_m:T
>assignvariableop_61_adam_multi_head_attention_1_query_kernel_m:N
<assignvariableop_62_adam_multi_head_attention_1_query_bias_m:R
<assignvariableop_63_adam_multi_head_attention_1_key_kernel_m:L
:assignvariableop_64_adam_multi_head_attention_1_key_bias_m:T
>assignvariableop_65_adam_multi_head_attention_1_value_kernel_m:N
<assignvariableop_66_adam_multi_head_attention_1_value_bias_m:_
Iassignvariableop_67_adam_multi_head_attention_1_attention_output_kernel_m:U
Gassignvariableop_68_adam_multi_head_attention_1_attention_output_bias_m:>
(assignvariableop_69_adam_conv1d_kernel_v:@4
&assignvariableop_70_adam_conv1d_bias_v:@@
*assignvariableop_71_adam_conv1d_1_kernel_v:@6
(assignvariableop_72_adam_conv1d_1_bias_v:@
*assignvariableop_73_adam_conv1d_2_kernel_v:@6
(assignvariableop_74_adam_conv1d_2_bias_v:@@
*assignvariableop_75_adam_conv1d_3_kernel_v:@6
(assignvariableop_76_adam_conv1d_3_bias_v::
'assignvariableop_77_adam_dense_kernel_v:	ї@3
%assignvariableop_78_adam_dense_bias_v:@;
)assignvariableop_79_adam_dense_1_kernel_v:@@5
'assignvariableop_80_adam_dense_1_bias_v:@;
)assignvariableop_81_adam_dense_2_kernel_v:@5
'assignvariableop_82_adam_dense_2_bias_v:R
<assignvariableop_83_adam_multi_head_attention_query_kernel_v:L
:assignvariableop_84_adam_multi_head_attention_query_bias_v:P
:assignvariableop_85_adam_multi_head_attention_key_kernel_v:J
8assignvariableop_86_adam_multi_head_attention_key_bias_v:R
<assignvariableop_87_adam_multi_head_attention_value_kernel_v:L
:assignvariableop_88_adam_multi_head_attention_value_bias_v:]
Gassignvariableop_89_adam_multi_head_attention_attention_output_kernel_v:S
Eassignvariableop_90_adam_multi_head_attention_attention_output_bias_v:T
>assignvariableop_91_adam_multi_head_attention_1_query_kernel_v:N
<assignvariableop_92_adam_multi_head_attention_1_query_bias_v:R
<assignvariableop_93_adam_multi_head_attention_1_key_kernel_v:L
:assignvariableop_94_adam_multi_head_attention_1_key_bias_v:T
>assignvariableop_95_adam_multi_head_attention_1_value_kernel_v:N
<assignvariableop_96_adam_multi_head_attention_1_value_bias_v:_
Iassignvariableop_97_adam_multi_head_attention_1_attention_output_kernel_v:U
Gassignvariableop_98_adam_multi_head_attention_1_attention_output_bias_v:
identity_100ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_43ҐAssignVariableOp_44ҐAssignVariableOp_45ҐAssignVariableOp_46ҐAssignVariableOp_47ҐAssignVariableOp_48ҐAssignVariableOp_49ҐAssignVariableOp_5ҐAssignVariableOp_50ҐAssignVariableOp_51ҐAssignVariableOp_52ҐAssignVariableOp_53ҐAssignVariableOp_54ҐAssignVariableOp_55ҐAssignVariableOp_56ҐAssignVariableOp_57ҐAssignVariableOp_58ҐAssignVariableOp_59ҐAssignVariableOp_6ҐAssignVariableOp_60ҐAssignVariableOp_61ҐAssignVariableOp_62ҐAssignVariableOp_63ҐAssignVariableOp_64ҐAssignVariableOp_65ҐAssignVariableOp_66ҐAssignVariableOp_67ҐAssignVariableOp_68ҐAssignVariableOp_69ҐAssignVariableOp_7ҐAssignVariableOp_70ҐAssignVariableOp_71ҐAssignVariableOp_72ҐAssignVariableOp_73ҐAssignVariableOp_74ҐAssignVariableOp_75ҐAssignVariableOp_76ҐAssignVariableOp_77ҐAssignVariableOp_78ҐAssignVariableOp_79ҐAssignVariableOp_8ҐAssignVariableOp_80ҐAssignVariableOp_81ҐAssignVariableOp_82ҐAssignVariableOp_83ҐAssignVariableOp_84ҐAssignVariableOp_85ҐAssignVariableOp_86ҐAssignVariableOp_87ҐAssignVariableOp_88ҐAssignVariableOp_89ҐAssignVariableOp_9ҐAssignVariableOp_90ҐAssignVariableOp_91ҐAssignVariableOp_92ҐAssignVariableOp_93ҐAssignVariableOp_94ҐAssignVariableOp_95ҐAssignVariableOp_96ҐAssignVariableOp_97ҐAssignVariableOp_98ш2
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:d*
dtype0*Ю2
valueФ2BС2dB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHї
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:d*
dtype0*Ё
value”B–dB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Х
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*¶
_output_shapesУ
Р::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*r
dtypesh
f2d	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOpAssignVariableOpassignvariableop_conv1d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv1d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv1d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv1d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv1d_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv1d_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv1d_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv1d_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_8AssignVariableOpassignvariableop_8_dense_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_1_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_1_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_2_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_2_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0	*
_output_shapes
:О
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_iterIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_beta_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_beta_2Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_decayIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_learning_rateIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_19AssignVariableOp5assignvariableop_19_multi_head_attention_query_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_20AssignVariableOp3assignvariableop_20_multi_head_attention_query_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_21AssignVariableOp3assignvariableop_21_multi_head_attention_key_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Ґ
AssignVariableOp_22AssignVariableOp1assignvariableop_22_multi_head_attention_key_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_23AssignVariableOp5assignvariableop_23_multi_head_attention_value_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_24AssignVariableOp3assignvariableop_24_multi_head_attention_value_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_25AssignVariableOp@assignvariableop_25_multi_head_attention_attention_output_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:ѓ
AssignVariableOp_26AssignVariableOp>assignvariableop_26_multi_head_attention_attention_output_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_27AssignVariableOp7assignvariableop_27_multi_head_attention_1_query_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_28AssignVariableOp5assignvariableop_28_multi_head_attention_1_query_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_29AssignVariableOp5assignvariableop_29_multi_head_attention_1_key_kernelIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_30AssignVariableOp3assignvariableop_30_multi_head_attention_1_key_biasIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_31AssignVariableOp7assignvariableop_31_multi_head_attention_1_value_kernelIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_32AssignVariableOp5assignvariableop_32_multi_head_attention_1_value_biasIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:≥
AssignVariableOp_33AssignVariableOpBassignvariableop_33_multi_head_attention_1_attention_output_kernelIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_34AssignVariableOp@assignvariableop_34_multi_head_attention_1_attention_output_biasIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_35AssignVariableOpassignvariableop_35_totalIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_36AssignVariableOpassignvariableop_36_countIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_37AssignVariableOpassignvariableop_37_total_1Identity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_38AssignVariableOpassignvariableop_38_count_1Identity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_39AssignVariableOp(assignvariableop_39_adam_conv1d_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_40AssignVariableOp&assignvariableop_40_adam_conv1d_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_conv1d_1_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_conv1d_1_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_conv1d_2_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_conv1d_2_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_conv1d_3_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_conv1d_3_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_47AssignVariableOp'assignvariableop_47_adam_dense_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_48AssignVariableOp%assignvariableop_48_adam_dense_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_49AssignVariableOp)assignvariableop_49_adam_dense_1_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_50AssignVariableOp'assignvariableop_50_adam_dense_1_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_51AssignVariableOp)assignvariableop_51_adam_dense_2_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_52AssignVariableOp'assignvariableop_52_adam_dense_2_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:≠
AssignVariableOp_53AssignVariableOp<assignvariableop_53_adam_multi_head_attention_query_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_54AssignVariableOp:assignvariableop_54_adam_multi_head_attention_query_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_55AssignVariableOp:assignvariableop_55_adam_multi_head_attention_key_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_56AssignVariableOp8assignvariableop_56_adam_multi_head_attention_key_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:≠
AssignVariableOp_57AssignVariableOp<assignvariableop_57_adam_multi_head_attention_value_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_58AssignVariableOp:assignvariableop_58_adam_multi_head_attention_value_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_59AssignVariableOpGassignvariableop_59_adam_multi_head_attention_attention_output_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:ґ
AssignVariableOp_60AssignVariableOpEassignvariableop_60_adam_multi_head_attention_attention_output_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:ѓ
AssignVariableOp_61AssignVariableOp>assignvariableop_61_adam_multi_head_attention_1_query_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:≠
AssignVariableOp_62AssignVariableOp<assignvariableop_62_adam_multi_head_attention_1_query_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:≠
AssignVariableOp_63AssignVariableOp<assignvariableop_63_adam_multi_head_attention_1_key_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_64AssignVariableOp:assignvariableop_64_adam_multi_head_attention_1_key_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:ѓ
AssignVariableOp_65AssignVariableOp>assignvariableop_65_adam_multi_head_attention_1_value_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:≠
AssignVariableOp_66AssignVariableOp<assignvariableop_66_adam_multi_head_attention_1_value_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_67AssignVariableOpIassignvariableop_67_adam_multi_head_attention_1_attention_output_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_68AssignVariableOpGassignvariableop_68_adam_multi_head_attention_1_attention_output_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_69AssignVariableOp(assignvariableop_69_adam_conv1d_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_70AssignVariableOp&assignvariableop_70_adam_conv1d_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_71AssignVariableOp*assignvariableop_71_adam_conv1d_1_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_72AssignVariableOp(assignvariableop_72_adam_conv1d_1_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_73AssignVariableOp*assignvariableop_73_adam_conv1d_2_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_74AssignVariableOp(assignvariableop_74_adam_conv1d_2_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_75AssignVariableOp*assignvariableop_75_adam_conv1d_3_kernel_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_76AssignVariableOp(assignvariableop_76_adam_conv1d_3_bias_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_77AssignVariableOp'assignvariableop_77_adam_dense_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_78AssignVariableOp%assignvariableop_78_adam_dense_bias_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_79AssignVariableOp)assignvariableop_79_adam_dense_1_kernel_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_80AssignVariableOp'assignvariableop_80_adam_dense_1_bias_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_81AssignVariableOp)assignvariableop_81_adam_dense_2_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_82AssignVariableOp'assignvariableop_82_adam_dense_2_bias_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:≠
AssignVariableOp_83AssignVariableOp<assignvariableop_83_adam_multi_head_attention_query_kernel_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_84AssignVariableOp:assignvariableop_84_adam_multi_head_attention_query_bias_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_85AssignVariableOp:assignvariableop_85_adam_multi_head_attention_key_kernel_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_86AssignVariableOp8assignvariableop_86_adam_multi_head_attention_key_bias_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:≠
AssignVariableOp_87AssignVariableOp<assignvariableop_87_adam_multi_head_attention_value_kernel_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_88AssignVariableOp:assignvariableop_88_adam_multi_head_attention_value_bias_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_89AssignVariableOpGassignvariableop_89_adam_multi_head_attention_attention_output_kernel_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:ґ
AssignVariableOp_90AssignVariableOpEassignvariableop_90_adam_multi_head_attention_attention_output_bias_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:ѓ
AssignVariableOp_91AssignVariableOp>assignvariableop_91_adam_multi_head_attention_1_query_kernel_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:≠
AssignVariableOp_92AssignVariableOp<assignvariableop_92_adam_multi_head_attention_1_query_bias_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:≠
AssignVariableOp_93AssignVariableOp<assignvariableop_93_adam_multi_head_attention_1_key_kernel_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_94AssignVariableOp:assignvariableop_94_adam_multi_head_attention_1_key_bias_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:ѓ
AssignVariableOp_95AssignVariableOp>assignvariableop_95_adam_multi_head_attention_1_value_kernel_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:≠
AssignVariableOp_96AssignVariableOp<assignvariableop_96_adam_multi_head_attention_1_value_bias_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_97AssignVariableOpIassignvariableop_97_adam_multi_head_attention_1_attention_output_kernel_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_98AssignVariableOpGassignvariableop_98_adam_multi_head_attention_1_attention_output_bias_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 —
Identity_99Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^NoOp"/device:CPU:0*
T0*
_output_shapes
: X
Identity_100IdentityIdentity_99:output:0^NoOp_1*
T0*
_output_shapes
: Њ
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98*"
_acd_function_control_output(*
_output_shapes
 "%
identity_100Identity_100:output:0*Ё
_input_shapesЋ
»: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_98:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
–
У
D__inference_conv1d_1_layer_call_and_return_conditional_losses_189967

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐ"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€В
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ї@Т
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : †
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@Ѓ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ї*
paddingVALID*
strides
Б
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї*
squeeze_dims

э€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0В
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€їU
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€їf
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€їД
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€ї@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€ї@
 
_user_specified_nameinputs
Г
l
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_188151

inputs
identityW
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :m
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€]
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Я

ф
C__inference_dense_2_layer_call_and_return_conditional_losses_190263

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
”
д
&__inference_model_layer_call_fn_188496
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10: 

unknown_11:

unknown_12: 

unknown_13:

unknown_14: 

unknown_15:

unknown_16: 

unknown_17:

unknown_18: 

unknown_19:@

unknown_20:@ 

unknown_21:@

unknown_22:

unknown_23:	ї@

unknown_24:@

unknown_25:@@

unknown_26:@

unknown_27:@

unknown_28:
identityИҐStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*@
_read_only_resource_inputs"
 	
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_188433o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:€€€€€€€€€ї: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:€€€€€€€€€ї
!
_user_specified_name	input_1
≤*
щ
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_189848	
query	
valueA
+query_einsum_einsum_readvariableop_resource:3
!query_add_readvariableop_resource:?
)key_einsum_einsum_readvariableop_resource:1
key_add_readvariableop_resource:A
+value_einsum_einsum_readvariableop_resource:3
!value_add_readvariableop_resource:L
6attention_output_einsum_einsum_readvariableop_resource::
,attention_output_add_readvariableop_resource:
identityИҐ#attention_output/add/ReadVariableOpҐ-attention_output/einsum/Einsum/ReadVariableOpҐkey/add/ReadVariableOpҐ key/einsum/Einsum/ReadVariableOpҐquery/add/ReadVariableOpҐ"query/einsum/Einsum/ReadVariableOpҐvalue/add/ReadVariableOpҐ"value/einsum/Einsum/ReadVariableOpТ
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0±
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:*
dtype0Н
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€їО
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0≠
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:*
dtype0З
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€їТ
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0±
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:*
dtype0Н
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€їJ
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *уµ>d
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€їС
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:€€€€€€€€€її*
equationaecd,abcd->acben
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€їїs
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*1
_output_shapes
:€€€€€€€€€її¶
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationacbe,aecd->abcd®
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0÷
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€ї*
equationabcd,cde->abeМ
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0™
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€їl
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€їЎ
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:€€€€€€€€€ї:€€€€€€€€€ї: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:€€€€€€€€€ї

_user_specified_namequery:SO
,
_output_shapes
:€€€€€€€€€ї

_user_specified_namevalue
≤*
щ
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_188196	
query	
valueA
+query_einsum_einsum_readvariableop_resource:3
!query_add_readvariableop_resource:?
)key_einsum_einsum_readvariableop_resource:1
key_add_readvariableop_resource:A
+value_einsum_einsum_readvariableop_resource:3
!value_add_readvariableop_resource:L
6attention_output_einsum_einsum_readvariableop_resource::
,attention_output_add_readvariableop_resource:
identityИҐ#attention_output/add/ReadVariableOpҐ-attention_output/einsum/Einsum/ReadVariableOpҐkey/add/ReadVariableOpҐ key/einsum/Einsum/ReadVariableOpҐquery/add/ReadVariableOpҐ"query/einsum/Einsum/ReadVariableOpҐvalue/add/ReadVariableOpҐ"value/einsum/Einsum/ReadVariableOpТ
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0±
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:*
dtype0Н
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€їО
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0≠
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:*
dtype0З
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€їТ
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0±
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:*
dtype0Н
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€їJ
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *уµ>d
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€їС
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:€€€€€€€€€її*
equationaecd,abcd->acben
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€їїs
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*1
_output_shapes
:€€€€€€€€€її¶
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationacbe,aecd->abcd®
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0÷
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€ї*
equationabcd,cde->abeМ
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0™
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€їl
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€їЎ
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:€€€€€€€€€ї:€€€€€€€€€ї: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:€€€€€€€€€ї

_user_specified_namequery:SO
,
_output_shapes
:€€€€€€€€€ї

_user_specified_namevalue
гш
ч
A__inference_model_layer_call_and_return_conditional_losses_189702

inputsV
@multi_head_attention_query_einsum_einsum_readvariableop_resource:H
6multi_head_attention_query_add_readvariableop_resource:T
>multi_head_attention_key_einsum_einsum_readvariableop_resource:F
4multi_head_attention_key_add_readvariableop_resource:V
@multi_head_attention_value_einsum_einsum_readvariableop_resource:H
6multi_head_attention_value_add_readvariableop_resource:a
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource:O
Amulti_head_attention_attention_output_add_readvariableop_resource:H
2conv1d_conv1d_expanddims_1_readvariableop_resource:@4
&conv1d_biasadd_readvariableop_resource:@J
4conv1d_1_conv1d_expanddims_1_readvariableop_resource:@6
(conv1d_1_biasadd_readvariableop_resource:X
Bmulti_head_attention_1_query_einsum_einsum_readvariableop_resource:J
8multi_head_attention_1_query_add_readvariableop_resource:V
@multi_head_attention_1_key_einsum_einsum_readvariableop_resource:H
6multi_head_attention_1_key_add_readvariableop_resource:X
Bmulti_head_attention_1_value_einsum_einsum_readvariableop_resource:J
8multi_head_attention_1_value_add_readvariableop_resource:c
Mmulti_head_attention_1_attention_output_einsum_einsum_readvariableop_resource:Q
Cmulti_head_attention_1_attention_output_add_readvariableop_resource:J
4conv1d_2_conv1d_expanddims_1_readvariableop_resource:@6
(conv1d_2_biasadd_readvariableop_resource:@J
4conv1d_3_conv1d_expanddims_1_readvariableop_resource:@6
(conv1d_3_biasadd_readvariableop_resource:7
$dense_matmul_readvariableop_resource:	ї@3
%dense_biasadd_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@@5
'dense_1_biasadd_readvariableop_resource:@8
&dense_2_matmul_readvariableop_resource:@5
'dense_2_biasadd_readvariableop_resource:
identityИҐconv1d/BiasAdd/ReadVariableOpҐ)conv1d/Conv1D/ExpandDims_1/ReadVariableOpҐconv1d_1/BiasAdd/ReadVariableOpҐ+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpҐconv1d_2/BiasAdd/ReadVariableOpҐ+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpҐconv1d_3/BiasAdd/ReadVariableOpҐ+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOpҐdense/BiasAdd/ReadVariableOpҐdense/MatMul/ReadVariableOpҐdense_1/BiasAdd/ReadVariableOpҐdense_1/MatMul/ReadVariableOpҐdense_2/BiasAdd/ReadVariableOpҐdense_2/MatMul/ReadVariableOpҐ8multi_head_attention/attention_output/add/ReadVariableOpҐBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpҐ+multi_head_attention/key/add/ReadVariableOpҐ5multi_head_attention/key/einsum/Einsum/ReadVariableOpҐ-multi_head_attention/query/add/ReadVariableOpҐ7multi_head_attention/query/einsum/Einsum/ReadVariableOpҐ-multi_head_attention/value/add/ReadVariableOpҐ7multi_head_attention/value/einsum/Einsum/ReadVariableOpҐ:multi_head_attention_1/attention_output/add/ReadVariableOpҐDmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpҐ-multi_head_attention_1/key/add/ReadVariableOpҐ7multi_head_attention_1/key/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_1/query/add/ReadVariableOpҐ9multi_head_attention_1/query/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_1/value/add/ReadVariableOpҐ9multi_head_attention_1/value/einsum/Einsum/ReadVariableOpЉ
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0№
(multi_head_attention/query/einsum/EinsumEinsuminputs?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationabc,cde->abde§
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0ћ
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€їЄ
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0Ў
&multi_head_attention/key/einsum/EinsumEinsuminputs=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationabc,cde->abde†
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0∆
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€їЉ
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0№
(multi_head_attention/value/einsum/EinsumEinsuminputs?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationabc,cde->abde§
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0ћ
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ї_
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *уµ>£
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€ї–
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*1
_output_shapes
:€€€€€€€€€її*
equationaecd,abcd->acbeШ
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€їїo
*multi_head_attention/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?–
(multi_head_attention/dropout/dropout/MulMul.multi_head_attention/softmax/Softmax:softmax:03multi_head_attention/dropout/dropout/Const:output:0*
T0*1
_output_shapes
:€€€€€€€€€їїИ
*multi_head_attention/dropout/dropout/ShapeShape.multi_head_attention/softmax/Softmax:softmax:0*
T0*
_output_shapes
:б
Amulti_head_attention/dropout/dropout/random_uniform/RandomUniformRandomUniform3multi_head_attention/dropout/dropout/Shape:output:0*
T0*1
_output_shapes
:€€€€€€€€€її*
dtype0*
seed2€€€€x
3multi_head_attention/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=З
1multi_head_attention/dropout/dropout/GreaterEqualGreaterEqualJmulti_head_attention/dropout/dropout/random_uniform/RandomUniform:output:0<multi_head_attention/dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:€€€€€€€€€її≥
)multi_head_attention/dropout/dropout/CastCast5multi_head_attention/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:€€€€€€€€€її 
*multi_head_attention/dropout/dropout/Mul_1Mul,multi_head_attention/dropout/dropout/Mul:z:0-multi_head_attention/dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:€€€€€€€€€їїе
$multi_head_attention/einsum_1/EinsumEinsum.multi_head_attention/dropout/dropout/Mul_1:z:0"multi_head_attention/value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationacbe,aecd->abcd“
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0Х
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€ї*
equationabcd,cde->abeґ
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0й
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€їС
tf.__operators__.add/AddV2AddV2-multi_head_attention/attention_output/add:z:0inputs*
T0*,
_output_shapes
:€€€€€€€€€їg
conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€®
conv1d/Conv1D/ExpandDims
ExpandDimstf.__operators__.add/AddV2:z:0%conv1d/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ї†
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0`
conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : µ
conv1d/Conv1D/ExpandDims_1
ExpandDims1conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@√
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ї@*
paddingVALID*
strides
П
conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї@*
squeeze_dims

э€€€€€€€€А
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ч
conv1d/BiasAddBiasAddconv1d/Conv1D/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€ї@c
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї@Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?М
dropout/dropout/MulMulconv1d/Relu:activations:0dropout/dropout/Const:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї@^
dropout/dropout/ShapeShapeconv1d/Relu:activations:0*
T0*
_output_shapes
:Ѓ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї@*
dtype0*
seed2c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=√
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї@Д
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:€€€€€€€€€ї@Ж
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*,
_output_shapes
:€€€€€€€€€ї@i
conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€І
conv1d_1/Conv1D/ExpandDims
ExpandDimsdropout/dropout/Mul_1:z:0'conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ї@§
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0b
 conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ї
conv1d_1/Conv1D/ExpandDims_1
ExpandDims3conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@…
conv1d_1/Conv1DConv2D#conv1d_1/Conv1D/ExpandDims:output:0%conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ї*
paddingVALID*
strides
У
conv1d_1/Conv1D/SqueezeSqueezeconv1d_1/Conv1D:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї*
squeeze_dims

э€€€€€€€€Д
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Э
conv1d_1/BiasAddBiasAdd conv1d_1/Conv1D/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€їg
conv1d_1/ReluReluconv1d_1/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€їЩ
tf.__operators__.add_1/AddV2AddV2conv1d_1/Relu:activations:0tf.__operators__.add/AddV2:z:0*
T0*,
_output_shapes
:€€€€€€€€€їј
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_1_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0ъ
*multi_head_attention_1/query/einsum/EinsumEinsum tf.__operators__.add_1/AddV2:z:0Amulti_head_attention_1/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationabc,cde->abde®
/multi_head_attention_1/query/add/ReadVariableOpReadVariableOp8multi_head_attention_1_query_add_readvariableop_resource*
_output_shapes

:*
dtype0“
 multi_head_attention_1/query/addAddV23multi_head_attention_1/query/einsum/Einsum:output:07multi_head_attention_1/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€їЉ
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_1_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0ц
(multi_head_attention_1/key/einsum/EinsumEinsum tf.__operators__.add_1/AddV2:z:0?multi_head_attention_1/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationabc,cde->abde§
-multi_head_attention_1/key/add/ReadVariableOpReadVariableOp6multi_head_attention_1_key_add_readvariableop_resource*
_output_shapes

:*
dtype0ћ
multi_head_attention_1/key/addAddV21multi_head_attention_1/key/einsum/Einsum:output:05multi_head_attention_1/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€їј
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_1_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0ъ
*multi_head_attention_1/value/einsum/EinsumEinsum tf.__operators__.add_1/AddV2:z:0Amulti_head_attention_1/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationabc,cde->abde®
/multi_head_attention_1/value/add/ReadVariableOpReadVariableOp8multi_head_attention_1_value_add_readvariableop_resource*
_output_shapes

:*
dtype0“
 multi_head_attention_1/value/addAddV23multi_head_attention_1/value/einsum/Einsum:output:07multi_head_attention_1/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€їa
multi_head_attention_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *уµ>©
multi_head_attention_1/MulMul$multi_head_attention_1/query/add:z:0%multi_head_attention_1/Mul/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€ї÷
$multi_head_attention_1/einsum/EinsumEinsum"multi_head_attention_1/key/add:z:0multi_head_attention_1/Mul:z:0*
N*
T0*1
_output_shapes
:€€€€€€€€€її*
equationaecd,abcd->acbeЬ
&multi_head_attention_1/softmax/SoftmaxSoftmax-multi_head_attention_1/einsum/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€їїq
,multi_head_attention_1/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?÷
*multi_head_attention_1/dropout/dropout/MulMul0multi_head_attention_1/softmax/Softmax:softmax:05multi_head_attention_1/dropout/dropout/Const:output:0*
T0*1
_output_shapes
:€€€€€€€€€їїМ
,multi_head_attention_1/dropout/dropout/ShapeShape0multi_head_attention_1/softmax/Softmax:softmax:0*
T0*
_output_shapes
:б
Cmulti_head_attention_1/dropout/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention_1/dropout/dropout/Shape:output:0*
T0*1
_output_shapes
:€€€€€€€€€її*
dtype0*
seed2z
5multi_head_attention_1/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=Н
3multi_head_attention_1/dropout/dropout/GreaterEqualGreaterEqualLmulti_head_attention_1/dropout/dropout/random_uniform/RandomUniform:output:0>multi_head_attention_1/dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:€€€€€€€€€їїЈ
+multi_head_attention_1/dropout/dropout/CastCast7multi_head_attention_1/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:€€€€€€€€€її–
,multi_head_attention_1/dropout/dropout/Mul_1Mul.multi_head_attention_1/dropout/dropout/Mul:z:0/multi_head_attention_1/dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:€€€€€€€€€їїл
&multi_head_attention_1/einsum_1/EinsumEinsum0multi_head_attention_1/dropout/dropout/Mul_1:z:0$multi_head_attention_1/value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationacbe,aecd->abcd÷
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_1_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0Ы
5multi_head_attention_1/attention_output/einsum/EinsumEinsum/multi_head_attention_1/einsum_1/Einsum:output:0Lmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€ї*
equationabcd,cde->abeЇ
:multi_head_attention_1/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_1_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0п
+multi_head_attention_1/attention_output/addAddV2>multi_head_attention_1/attention_output/einsum/Einsum:output:0Bmulti_head_attention_1/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€їѓ
tf.__operators__.add_2/AddV2AddV2/multi_head_attention_1/attention_output/add:z:0 tf.__operators__.add_1/AddV2:z:0*
T0*,
_output_shapes
:€€€€€€€€€їi
conv1d_2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€Ѓ
conv1d_2/Conv1D/ExpandDims
ExpandDims tf.__operators__.add_2/AddV2:z:0'conv1d_2/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ї§
+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0b
 conv1d_2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ї
conv1d_2/Conv1D/ExpandDims_1
ExpandDims3conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@…
conv1d_2/Conv1DConv2D#conv1d_2/Conv1D/ExpandDims:output:0%conv1d_2/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ї@*
paddingVALID*
strides
У
conv1d_2/Conv1D/SqueezeSqueezeconv1d_2/Conv1D:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї@*
squeeze_dims

э€€€€€€€€Д
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Э
conv1d_2/BiasAddBiasAdd conv1d_2/Conv1D/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€ї@g
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї@\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?Т
dropout_1/dropout/MulMulconv1d_2/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї@b
dropout_1/dropout/ShapeShapeconv1d_2/Relu:activations:0*
T0*
_output_shapes
:≤
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї@*
dtype0*
seed2e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=…
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї@И
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:€€€€€€€€€ї@М
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*,
_output_shapes
:€€€€€€€€€ї@i
conv1d_3/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€©
conv1d_3/Conv1D/ExpandDims
ExpandDimsdropout_1/dropout/Mul_1:z:0'conv1d_3/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ї@§
+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0b
 conv1d_3/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ї
conv1d_3/Conv1D/ExpandDims_1
ExpandDims3conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_3/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@…
conv1d_3/Conv1DConv2D#conv1d_3/Conv1D/ExpandDims:output:0%conv1d_3/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ї*
paddingVALID*
strides
У
conv1d_3/Conv1D/SqueezeSqueezeconv1d_3/Conv1D:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї*
squeeze_dims

э€€€€€€€€Д
conv1d_3/BiasAdd/ReadVariableOpReadVariableOp(conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Э
conv1d_3/BiasAddBiasAdd conv1d_3/Conv1D/Squeeze:output:0'conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€їg
conv1d_3/ReluReluconv1d_3/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€їЫ
tf.__operators__.add_3/AddV2AddV2conv1d_3/Relu:activations:0 tf.__operators__.add_2/AddV2:z:0*
T0*,
_output_shapes
:€€€€€€€€€їl
*global_max_pooling1d/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :©
global_max_pooling1d/MaxMax tf.__operators__.add_3/AddV2:z:03global_max_pooling1d/Max/reduction_indices:output:0*
T0*(
_output_shapes
:€€€€€€€€€ї\
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?Ф
dropout_2/dropout/MulMul!global_max_pooling1d/Max:output:0 dropout_2/dropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€їh
dropout_2/dropout/ShapeShape!global_max_pooling1d/Max:output:0*
T0*
_output_shapes
:Ѓ
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ї*
dtype0*
seed2e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>≈
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€їД
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€їИ
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€їБ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	ї@*
dtype0К
dense/MatMulMatMuldropout_2/dropout/Mul_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0И
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Д
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0Л
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@В
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0О
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Д
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Н
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€В
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€f
dense_2/SoftmaxSoftmaxdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€h
IdentityIdentitydense_2/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€≈
NoOpNoOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_2/BiasAdd/ReadVariableOp,^conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_3/BiasAdd/ReadVariableOp,^conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp;^multi_head_attention_1/attention_output/add/ReadVariableOpE^multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_1/key/add/ReadVariableOp8^multi_head_attention_1/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_1/query/add/ReadVariableOp:^multi_head_attention_1/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_1/value/add/ReadVariableOp:^multi_head_attention_1/value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:€€€€€€€€€ї: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/Conv1D/ExpandDims_1/ReadVariableOp)conv1d/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp2Z
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_2/BiasAdd/ReadVariableOpconv1d_2/BiasAdd/ReadVariableOp2Z
+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_3/BiasAdd/ReadVariableOpconv1d_3/BiasAdd/ReadVariableOp2Z
+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2t
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2И
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Z
+multi_head_attention/key/add/ReadVariableOp+multi_head_attention/key/add/ReadVariableOp2n
5multi_head_attention/key/einsum/Einsum/ReadVariableOp5multi_head_attention/key/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/query/add/ReadVariableOp-multi_head_attention/query/add/ReadVariableOp2r
7multi_head_attention/query/einsum/Einsum/ReadVariableOp7multi_head_attention/query/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/value/add/ReadVariableOp-multi_head_attention/value/add/ReadVariableOp2r
7multi_head_attention/value/einsum/Einsum/ReadVariableOp7multi_head_attention/value/einsum/Einsum/ReadVariableOp2x
:multi_head_attention_1/attention_output/add/ReadVariableOp:multi_head_attention_1/attention_output/add/ReadVariableOp2М
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_1/key/add/ReadVariableOp-multi_head_attention_1/key/add/ReadVariableOp2r
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOp7multi_head_attention_1/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_1/query/add/ReadVariableOp/multi_head_attention_1/query/add/ReadVariableOp2v
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_1/value/add/ReadVariableOp/multi_head_attention_1/value/add/ReadVariableOp2v
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€ї
 
_user_specified_nameinputs
Д
a
(__inference_dropout_layer_call_fn_189925

inputs
identityИҐStatefulPartitionedCall∆
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ї@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_188709t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ї@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ї@22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€ї@
 
_user_specified_nameinputs
Ж
к
5__inference_multi_head_attention_layer_call_fn_189813	
query	
value
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identityИҐStatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ї**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_188790t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ї`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:€€€€€€€€€ї:€€€€€€€€€ї: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
,
_output_shapes
:€€€€€€€€€ї

_user_specified_namequery:SO
,
_output_shapes
:€€€€€€€€€ї

_user_specified_namevalue
К
м
7__inference_multi_head_attention_1_layer_call_fn_190011	
query	
value
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identityИҐStatefulPartitionedCallƒ
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ї**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_188660t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ї`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:€€€€€€€€€ї:€€€€€€€€€ї: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
,
_output_shapes
:€€€€€€€€€ї

_user_specified_namequery:SO
,
_output_shapes
:€€€€€€€€€ї

_user_specified_namevalue
ц
Q
5__inference_global_max_pooling1d_layer_call_fn_190170

inputs
identity«
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_188151i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
–
г
&__inference_model_layer_call_fn_189383

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10: 

unknown_11:

unknown_12: 

unknown_13:

unknown_14: 

unknown_15:

unknown_16: 

unknown_17:

unknown_18: 

unknown_19:@

unknown_20:@ 

unknown_21:@

unknown_22:

unknown_23:	ї@

unknown_24:@

unknown_25:@@

unknown_26:@

unknown_27:@

unknown_28:
identityИҐStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*@
_read_only_resource_inputs"
 	
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_188957o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:€€€€€€€€€ї: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€ї
 
_user_specified_nameinputs
єI
Ч
A__inference_model_layer_call_and_return_conditional_losses_188433

inputs1
multi_head_attention_188197:-
multi_head_attention_188199:1
multi_head_attention_188201:-
multi_head_attention_188203:1
multi_head_attention_188205:-
multi_head_attention_188207:1
multi_head_attention_188209:)
multi_head_attention_188211:#
conv1d_188232:@
conv1d_188234:@%
conv1d_1_188261:@
conv1d_1_188263:3
multi_head_attention_1_188303:/
multi_head_attention_1_188305:3
multi_head_attention_1_188307:/
multi_head_attention_1_188309:3
multi_head_attention_1_188311:/
multi_head_attention_1_188313:3
multi_head_attention_1_188315:+
multi_head_attention_1_188317:%
conv1d_2_188338:@
conv1d_2_188340:@%
conv1d_3_188367:@
conv1d_3_188369:
dense_188393:	ї@
dense_188395:@ 
dense_1_188410:@@
dense_1_188412:@ 
dense_2_188427:@
dense_2_188429:
identityИҐconv1d/StatefulPartitionedCallҐ conv1d_1/StatefulPartitionedCallҐ conv1d_2/StatefulPartitionedCallҐ conv1d_3/StatefulPartitionedCallҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐdense_2/StatefulPartitionedCallҐ,multi_head_attention/StatefulPartitionedCallҐ.multi_head_attention_1/StatefulPartitionedCallл
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCallinputsinputsmulti_head_attention_188197multi_head_attention_188199multi_head_attention_188201multi_head_attention_188203multi_head_attention_188205multi_head_attention_188207multi_head_attention_188209multi_head_attention_188211*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ї**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_188196Щ
tf.__operators__.add/AddV2AddV25multi_head_attention/StatefulPartitionedCall:output:0inputs*
T0*,
_output_shapes
:€€€€€€€€€їИ
conv1d/StatefulPartitionedCallStatefulPartitionedCalltf.__operators__.add/AddV2:z:0conv1d_188232conv1d_188234*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ї@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_188231я
dropout/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ї@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_188242Т
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0conv1d_1_188261conv1d_1_188263*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ї*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_188260І
tf.__operators__.add_1/AddV2AddV2)conv1d_1/StatefulPartitionedCall:output:0tf.__operators__.add/AddV2:z:0*
T0*,
_output_shapes
:€€€€€€€€€ї≥
.multi_head_attention_1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_1/AddV2:z:0 tf.__operators__.add_1/AddV2:z:0multi_head_attention_1_188303multi_head_attention_1_188305multi_head_attention_1_188307multi_head_attention_1_188309multi_head_attention_1_188311multi_head_attention_1_188313multi_head_attention_1_188315multi_head_attention_1_188317*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ї**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_188302Ј
tf.__operators__.add_2/AddV2AddV27multi_head_attention_1/StatefulPartitionedCall:output:0 tf.__operators__.add_1/AddV2:z:0*
T0*,
_output_shapes
:€€€€€€€€€їТ
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_2/AddV2:z:0conv1d_2_188338conv1d_2_188340*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ї@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_188337е
dropout_1/PartitionedCallPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ї@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_188348Ф
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0conv1d_3_188367conv1d_3_188369*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ї*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv1d_3_layer_call_and_return_conditional_losses_188366©
tf.__operators__.add_3/AddV2AddV2)conv1d_3/StatefulPartitionedCall:output:0 tf.__operators__.add_2/AddV2:z:0*
T0*,
_output_shapes
:€€€€€€€€€їо
$global_max_pooling1d/PartitionedCallPartitionedCall tf.__operators__.add_3/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ї* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_188151е
dropout_2/PartitionedCallPartitionedCall-global_max_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ї* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_188379Г
dense/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_188393dense_188395*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_188392П
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_188410dense_1_188412*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_188409С
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_188427dense_2_188429*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_188426w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Ф
NoOpNoOp^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCall/^multi_head_attention_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:€€€€€€€€€ї: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2\
,multi_head_attention/StatefulPartitionedCall,multi_head_attention/StatefulPartitionedCall2`
.multi_head_attention_1/StatefulPartitionedCall.multi_head_attention_1/StatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€ї
 
_user_specified_nameinputs
≤
D
(__inference_dropout_layer_call_fn_189920

inputs
identityґ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ї@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_188242e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ї@:T P
,
_output_shapes
:€€€€€€€€€ї@
 
_user_specified_nameinputs
–
У
D__inference_conv1d_3_layer_call_and_return_conditional_losses_190165

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐ"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€В
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ї@Т
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : †
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@Ѓ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ї*
paddingVALID*
strides
Б
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї*
squeeze_dims

э€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0В
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€їU
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€їf
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€їД
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€ї@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€ї@
 
_user_specified_nameinputs
ая
ј
!__inference__wrapped_model_188141
input_1\
Fmodel_multi_head_attention_query_einsum_einsum_readvariableop_resource:N
<model_multi_head_attention_query_add_readvariableop_resource:Z
Dmodel_multi_head_attention_key_einsum_einsum_readvariableop_resource:L
:model_multi_head_attention_key_add_readvariableop_resource:\
Fmodel_multi_head_attention_value_einsum_einsum_readvariableop_resource:N
<model_multi_head_attention_value_add_readvariableop_resource:g
Qmodel_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource:U
Gmodel_multi_head_attention_attention_output_add_readvariableop_resource:N
8model_conv1d_conv1d_expanddims_1_readvariableop_resource:@:
,model_conv1d_biasadd_readvariableop_resource:@P
:model_conv1d_1_conv1d_expanddims_1_readvariableop_resource:@<
.model_conv1d_1_biasadd_readvariableop_resource:^
Hmodel_multi_head_attention_1_query_einsum_einsum_readvariableop_resource:P
>model_multi_head_attention_1_query_add_readvariableop_resource:\
Fmodel_multi_head_attention_1_key_einsum_einsum_readvariableop_resource:N
<model_multi_head_attention_1_key_add_readvariableop_resource:^
Hmodel_multi_head_attention_1_value_einsum_einsum_readvariableop_resource:P
>model_multi_head_attention_1_value_add_readvariableop_resource:i
Smodel_multi_head_attention_1_attention_output_einsum_einsum_readvariableop_resource:W
Imodel_multi_head_attention_1_attention_output_add_readvariableop_resource:P
:model_conv1d_2_conv1d_expanddims_1_readvariableop_resource:@<
.model_conv1d_2_biasadd_readvariableop_resource:@P
:model_conv1d_3_conv1d_expanddims_1_readvariableop_resource:@<
.model_conv1d_3_biasadd_readvariableop_resource:=
*model_dense_matmul_readvariableop_resource:	ї@9
+model_dense_biasadd_readvariableop_resource:@>
,model_dense_1_matmul_readvariableop_resource:@@;
-model_dense_1_biasadd_readvariableop_resource:@>
,model_dense_2_matmul_readvariableop_resource:@;
-model_dense_2_biasadd_readvariableop_resource:
identityИҐ#model/conv1d/BiasAdd/ReadVariableOpҐ/model/conv1d/Conv1D/ExpandDims_1/ReadVariableOpҐ%model/conv1d_1/BiasAdd/ReadVariableOpҐ1model/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpҐ%model/conv1d_2/BiasAdd/ReadVariableOpҐ1model/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpҐ%model/conv1d_3/BiasAdd/ReadVariableOpҐ1model/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOpҐ"model/dense/BiasAdd/ReadVariableOpҐ!model/dense/MatMul/ReadVariableOpҐ$model/dense_1/BiasAdd/ReadVariableOpҐ#model/dense_1/MatMul/ReadVariableOpҐ$model/dense_2/BiasAdd/ReadVariableOpҐ#model/dense_2/MatMul/ReadVariableOpҐ>model/multi_head_attention/attention_output/add/ReadVariableOpҐHmodel/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpҐ1model/multi_head_attention/key/add/ReadVariableOpҐ;model/multi_head_attention/key/einsum/Einsum/ReadVariableOpҐ3model/multi_head_attention/query/add/ReadVariableOpҐ=model/multi_head_attention/query/einsum/Einsum/ReadVariableOpҐ3model/multi_head_attention/value/add/ReadVariableOpҐ=model/multi_head_attention/value/einsum/Einsum/ReadVariableOpҐ@model/multi_head_attention_1/attention_output/add/ReadVariableOpҐJmodel/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpҐ3model/multi_head_attention_1/key/add/ReadVariableOpҐ=model/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpҐ5model/multi_head_attention_1/query/add/ReadVariableOpҐ?model/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpҐ5model/multi_head_attention_1/value/add/ReadVariableOpҐ?model/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp»
=model/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpFmodel_multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0й
.model/multi_head_attention/query/einsum/EinsumEinsuminput_1Emodel/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationabc,cde->abde∞
3model/multi_head_attention/query/add/ReadVariableOpReadVariableOp<model_multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0ё
$model/multi_head_attention/query/addAddV27model/multi_head_attention/query/einsum/Einsum:output:0;model/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€їƒ
;model/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpDmodel_multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0е
,model/multi_head_attention/key/einsum/EinsumEinsuminput_1Cmodel/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationabc,cde->abdeђ
1model/multi_head_attention/key/add/ReadVariableOpReadVariableOp:model_multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0Ў
"model/multi_head_attention/key/addAddV25model/multi_head_attention/key/einsum/Einsum:output:09model/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ї»
=model/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpFmodel_multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0й
.model/multi_head_attention/value/einsum/EinsumEinsuminput_1Emodel/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationabc,cde->abde∞
3model/multi_head_attention/value/add/ReadVariableOpReadVariableOp<model_multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0ё
$model/multi_head_attention/value/addAddV27model/multi_head_attention/value/einsum/Einsum:output:0;model/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€їe
 model/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *уµ>µ
model/multi_head_attention/MulMul(model/multi_head_attention/query/add:z:0)model/multi_head_attention/Mul/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€їв
(model/multi_head_attention/einsum/EinsumEinsum&model/multi_head_attention/key/add:z:0"model/multi_head_attention/Mul:z:0*
N*
T0*1
_output_shapes
:€€€€€€€€€її*
equationaecd,abcd->acbe§
*model/multi_head_attention/softmax/SoftmaxSoftmax1model/multi_head_attention/einsum/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€її©
+model/multi_head_attention/dropout/IdentityIdentity4model/multi_head_attention/softmax/Softmax:softmax:0*
T0*1
_output_shapes
:€€€€€€€€€їїч
*model/multi_head_attention/einsum_1/EinsumEinsum4model/multi_head_attention/dropout/Identity:output:0(model/multi_head_attention/value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationacbe,aecd->abcdё
Hmodel/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpQmodel_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0І
9model/multi_head_attention/attention_output/einsum/EinsumEinsum3model/multi_head_attention/einsum_1/Einsum:output:0Pmodel/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€ї*
equationabcd,cde->abe¬
>model/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpGmodel_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0ы
/model/multi_head_attention/attention_output/addAddV2Bmodel/multi_head_attention/attention_output/einsum/Einsum:output:0Fmodel/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€їЮ
 model/tf.__operators__.add/AddV2AddV23model/multi_head_attention/attention_output/add:z:0input_1*
T0*,
_output_shapes
:€€€€€€€€€їm
"model/conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€Ї
model/conv1d/Conv1D/ExpandDims
ExpandDims$model/tf.__operators__.add/AddV2:z:0+model/conv1d/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€їђ
/model/conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp8model_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0f
$model/conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : «
 model/conv1d/Conv1D/ExpandDims_1
ExpandDims7model/conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0-model/conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@’
model/conv1d/Conv1DConv2D'model/conv1d/Conv1D/ExpandDims:output:0)model/conv1d/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ї@*
paddingVALID*
strides
Ы
model/conv1d/Conv1D/SqueezeSqueezemodel/conv1d/Conv1D:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї@*
squeeze_dims

э€€€€€€€€М
#model/conv1d/BiasAdd/ReadVariableOpReadVariableOp,model_conv1d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0©
model/conv1d/BiasAddBiasAdd$model/conv1d/Conv1D/Squeeze:output:0+model/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€ї@o
model/conv1d/ReluRelumodel/conv1d/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї@z
model/dropout/IdentityIdentitymodel/conv1d/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€ї@o
$model/conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€є
 model/conv1d_1/Conv1D/ExpandDims
ExpandDimsmodel/dropout/Identity:output:0-model/conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ї@∞
1model/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0h
&model/conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Ќ
"model/conv1d_1/Conv1D/ExpandDims_1
ExpandDims9model/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0/model/conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@џ
model/conv1d_1/Conv1DConv2D)model/conv1d_1/Conv1D/ExpandDims:output:0+model/conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ї*
paddingVALID*
strides
Я
model/conv1d_1/Conv1D/SqueezeSqueezemodel/conv1d_1/Conv1D:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї*
squeeze_dims

э€€€€€€€€Р
%model/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ѓ
model/conv1d_1/BiasAddBiasAdd&model/conv1d_1/Conv1D/Squeeze:output:0-model/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€їs
model/conv1d_1/ReluRelumodel/conv1d_1/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€їЂ
"model/tf.__operators__.add_1/AddV2AddV2!model/conv1d_1/Relu:activations:0$model/tf.__operators__.add/AddV2:z:0*
T0*,
_output_shapes
:€€€€€€€€€їћ
?model/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpReadVariableOpHmodel_multi_head_attention_1_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0М
0model/multi_head_attention_1/query/einsum/EinsumEinsum&model/tf.__operators__.add_1/AddV2:z:0Gmodel/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationabc,cde->abdeі
5model/multi_head_attention_1/query/add/ReadVariableOpReadVariableOp>model_multi_head_attention_1_query_add_readvariableop_resource*
_output_shapes

:*
dtype0д
&model/multi_head_attention_1/query/addAddV29model/multi_head_attention_1/query/einsum/Einsum:output:0=model/multi_head_attention_1/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ї»
=model/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpReadVariableOpFmodel_multi_head_attention_1_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0И
.model/multi_head_attention_1/key/einsum/EinsumEinsum&model/tf.__operators__.add_1/AddV2:z:0Emodel/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationabc,cde->abde∞
3model/multi_head_attention_1/key/add/ReadVariableOpReadVariableOp<model_multi_head_attention_1_key_add_readvariableop_resource*
_output_shapes

:*
dtype0ё
$model/multi_head_attention_1/key/addAddV27model/multi_head_attention_1/key/einsum/Einsum:output:0;model/multi_head_attention_1/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€їћ
?model/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpReadVariableOpHmodel_multi_head_attention_1_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0М
0model/multi_head_attention_1/value/einsum/EinsumEinsum&model/tf.__operators__.add_1/AddV2:z:0Gmodel/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationabc,cde->abdeі
5model/multi_head_attention_1/value/add/ReadVariableOpReadVariableOp>model_multi_head_attention_1_value_add_readvariableop_resource*
_output_shapes

:*
dtype0д
&model/multi_head_attention_1/value/addAddV29model/multi_head_attention_1/value/einsum/Einsum:output:0=model/multi_head_attention_1/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€їg
"model/multi_head_attention_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *уµ>ї
 model/multi_head_attention_1/MulMul*model/multi_head_attention_1/query/add:z:0+model/multi_head_attention_1/Mul/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€їи
*model/multi_head_attention_1/einsum/EinsumEinsum(model/multi_head_attention_1/key/add:z:0$model/multi_head_attention_1/Mul:z:0*
N*
T0*1
_output_shapes
:€€€€€€€€€її*
equationaecd,abcd->acbe®
,model/multi_head_attention_1/softmax/SoftmaxSoftmax3model/multi_head_attention_1/einsum/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€її≠
-model/multi_head_attention_1/dropout/IdentityIdentity6model/multi_head_attention_1/softmax/Softmax:softmax:0*
T0*1
_output_shapes
:€€€€€€€€€їїэ
,model/multi_head_attention_1/einsum_1/EinsumEinsum6model/multi_head_attention_1/dropout/Identity:output:0*model/multi_head_attention_1/value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationacbe,aecd->abcdв
Jmodel/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpSmodel_multi_head_attention_1_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0≠
;model/multi_head_attention_1/attention_output/einsum/EinsumEinsum5model/multi_head_attention_1/einsum_1/Einsum:output:0Rmodel/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€ї*
equationabcd,cde->abe∆
@model/multi_head_attention_1/attention_output/add/ReadVariableOpReadVariableOpImodel_multi_head_attention_1_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0Б
1model/multi_head_attention_1/attention_output/addAddV2Dmodel/multi_head_attention_1/attention_output/einsum/Einsum:output:0Hmodel/multi_head_attention_1/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€їЅ
"model/tf.__operators__.add_2/AddV2AddV25model/multi_head_attention_1/attention_output/add:z:0&model/tf.__operators__.add_1/AddV2:z:0*
T0*,
_output_shapes
:€€€€€€€€€їo
$model/conv1d_2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€ј
 model/conv1d_2/Conv1D/ExpandDims
ExpandDims&model/tf.__operators__.add_2/AddV2:z:0-model/conv1d_2/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ї∞
1model/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0h
&model/conv1d_2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Ќ
"model/conv1d_2/Conv1D/ExpandDims_1
ExpandDims9model/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp:value:0/model/conv1d_2/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@џ
model/conv1d_2/Conv1DConv2D)model/conv1d_2/Conv1D/ExpandDims:output:0+model/conv1d_2/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ї@*
paddingVALID*
strides
Я
model/conv1d_2/Conv1D/SqueezeSqueezemodel/conv1d_2/Conv1D:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї@*
squeeze_dims

э€€€€€€€€Р
%model/conv1d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ѓ
model/conv1d_2/BiasAddBiasAdd&model/conv1d_2/Conv1D/Squeeze:output:0-model/conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€ї@s
model/conv1d_2/ReluRelumodel/conv1d_2/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї@~
model/dropout_1/IdentityIdentity!model/conv1d_2/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€ї@o
$model/conv1d_3/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€ї
 model/conv1d_3/Conv1D/ExpandDims
ExpandDims!model/dropout_1/Identity:output:0-model/conv1d_3/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ї@∞
1model/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0h
&model/conv1d_3/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Ќ
"model/conv1d_3/Conv1D/ExpandDims_1
ExpandDims9model/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp:value:0/model/conv1d_3/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@џ
model/conv1d_3/Conv1DConv2D)model/conv1d_3/Conv1D/ExpandDims:output:0+model/conv1d_3/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ї*
paddingVALID*
strides
Я
model/conv1d_3/Conv1D/SqueezeSqueezemodel/conv1d_3/Conv1D:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї*
squeeze_dims

э€€€€€€€€Р
%model/conv1d_3/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ѓ
model/conv1d_3/BiasAddBiasAdd&model/conv1d_3/Conv1D/Squeeze:output:0-model/conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€їs
model/conv1d_3/ReluRelumodel/conv1d_3/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї≠
"model/tf.__operators__.add_3/AddV2AddV2!model/conv1d_3/Relu:activations:0&model/tf.__operators__.add_2/AddV2:z:0*
T0*,
_output_shapes
:€€€€€€€€€їr
0model/global_max_pooling1d/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :ї
model/global_max_pooling1d/MaxMax&model/tf.__operators__.add_3/AddV2:z:09model/global_max_pooling1d/Max/reduction_indices:output:0*
T0*(
_output_shapes
:€€€€€€€€€їА
model/dropout_2/IdentityIdentity'model/global_max_pooling1d/Max:output:0*
T0*(
_output_shapes
:€€€€€€€€€їН
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes
:	ї@*
dtype0Ь
model/dense/MatMulMatMul!model/dropout_2/Identity:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@К
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ъ
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@h
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Р
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0Э
model/dense_1/MatMulMatMulmodel/dense/Relu:activations:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@О
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0†
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@l
model/dense_1/ReluRelumodel/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Р
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Я
model/dense_2/MatMulMatMul model/dense_1/Relu:activations:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€О
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0†
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
model/dense_2/SoftmaxSoftmaxmodel/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€n
IdentityIdentitymodel/dense_2/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€щ
NoOpNoOp$^model/conv1d/BiasAdd/ReadVariableOp0^model/conv1d/Conv1D/ExpandDims_1/ReadVariableOp&^model/conv1d_1/BiasAdd/ReadVariableOp2^model/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp&^model/conv1d_2/BiasAdd/ReadVariableOp2^model/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp&^model/conv1d_3/BiasAdd/ReadVariableOp2^model/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp?^model/multi_head_attention/attention_output/add/ReadVariableOpI^model/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2^model/multi_head_attention/key/add/ReadVariableOp<^model/multi_head_attention/key/einsum/Einsum/ReadVariableOp4^model/multi_head_attention/query/add/ReadVariableOp>^model/multi_head_attention/query/einsum/Einsum/ReadVariableOp4^model/multi_head_attention/value/add/ReadVariableOp>^model/multi_head_attention/value/einsum/Einsum/ReadVariableOpA^model/multi_head_attention_1/attention_output/add/ReadVariableOpK^model/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp4^model/multi_head_attention_1/key/add/ReadVariableOp>^model/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp6^model/multi_head_attention_1/query/add/ReadVariableOp@^model/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp6^model/multi_head_attention_1/value/add/ReadVariableOp@^model/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:€€€€€€€€€ї: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#model/conv1d/BiasAdd/ReadVariableOp#model/conv1d/BiasAdd/ReadVariableOp2b
/model/conv1d/Conv1D/ExpandDims_1/ReadVariableOp/model/conv1d/Conv1D/ExpandDims_1/ReadVariableOp2N
%model/conv1d_1/BiasAdd/ReadVariableOp%model/conv1d_1/BiasAdd/ReadVariableOp2f
1model/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp1model/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2N
%model/conv1d_2/BiasAdd/ReadVariableOp%model/conv1d_2/BiasAdd/ReadVariableOp2f
1model/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp1model/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp2N
%model/conv1d_3/BiasAdd/ReadVariableOp%model/conv1d_3/BiasAdd/ReadVariableOp2f
1model/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp1model/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2L
$model/dense_2/BiasAdd/ReadVariableOp$model/dense_2/BiasAdd/ReadVariableOp2J
#model/dense_2/MatMul/ReadVariableOp#model/dense_2/MatMul/ReadVariableOp2А
>model/multi_head_attention/attention_output/add/ReadVariableOp>model/multi_head_attention/attention_output/add/ReadVariableOp2Ф
Hmodel/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpHmodel/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2f
1model/multi_head_attention/key/add/ReadVariableOp1model/multi_head_attention/key/add/ReadVariableOp2z
;model/multi_head_attention/key/einsum/Einsum/ReadVariableOp;model/multi_head_attention/key/einsum/Einsum/ReadVariableOp2j
3model/multi_head_attention/query/add/ReadVariableOp3model/multi_head_attention/query/add/ReadVariableOp2~
=model/multi_head_attention/query/einsum/Einsum/ReadVariableOp=model/multi_head_attention/query/einsum/Einsum/ReadVariableOp2j
3model/multi_head_attention/value/add/ReadVariableOp3model/multi_head_attention/value/add/ReadVariableOp2~
=model/multi_head_attention/value/einsum/Einsum/ReadVariableOp=model/multi_head_attention/value/einsum/Einsum/ReadVariableOp2Д
@model/multi_head_attention_1/attention_output/add/ReadVariableOp@model/multi_head_attention_1/attention_output/add/ReadVariableOp2Ш
Jmodel/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpJmodel/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp2j
3model/multi_head_attention_1/key/add/ReadVariableOp3model/multi_head_attention_1/key/add/ReadVariableOp2~
=model/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp=model/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp2n
5model/multi_head_attention_1/query/add/ReadVariableOp5model/multi_head_attention_1/query/add/ReadVariableOp2В
?model/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp?model/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp2n
5model/multi_head_attention_1/value/add/ReadVariableOp5model/multi_head_attention_1/value/add/ReadVariableOp2В
?model/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp?model/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp:U Q
,
_output_shapes
:€€€€€€€€€ї
!
_user_specified_name	input_1
і*
ы
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_188302	
query	
valueA
+query_einsum_einsum_readvariableop_resource:3
!query_add_readvariableop_resource:?
)key_einsum_einsum_readvariableop_resource:1
key_add_readvariableop_resource:A
+value_einsum_einsum_readvariableop_resource:3
!value_add_readvariableop_resource:L
6attention_output_einsum_einsum_readvariableop_resource::
,attention_output_add_readvariableop_resource:
identityИҐ#attention_output/add/ReadVariableOpҐ-attention_output/einsum/Einsum/ReadVariableOpҐkey/add/ReadVariableOpҐ key/einsum/Einsum/ReadVariableOpҐquery/add/ReadVariableOpҐ"query/einsum/Einsum/ReadVariableOpҐvalue/add/ReadVariableOpҐ"value/einsum/Einsum/ReadVariableOpТ
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0±
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:*
dtype0Н
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€їО
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0≠
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:*
dtype0З
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€їТ
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0±
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:*
dtype0Н
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€їJ
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *уµ>d
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€їС
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:€€€€€€€€€її*
equationaecd,abcd->acben
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€їїs
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*1
_output_shapes
:€€€€€€€€€її¶
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationacbe,aecd->abcd®
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0÷
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€ї*
equationabcd,cde->abeМ
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0™
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€їl
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€їЎ
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:€€€€€€€€€ї:€€€€€€€€€ї: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:€€€€€€€€€ї

_user_specified_namequery:SO
,
_output_shapes
:€€€€€€€€€ї

_user_specified_namevalue
±
в
$__inference_signature_wrapper_189769
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10: 

unknown_11:

unknown_12: 

unknown_13:

unknown_14: 

unknown_15:

unknown_16: 

unknown_17:

unknown_18: 

unknown_19:@

unknown_20:@ 

unknown_21:@

unknown_22:

unknown_23:	ї@

unknown_24:@

unknown_25:@@

unknown_26:@

unknown_27:@

unknown_28:
identityИҐStatefulPartitionedCallє
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*@
_read_only_resource_inputs"
 	
*0
config_proto 

CPU

GPU2*0J 8В **
f%R#
!__inference__wrapped_model_188141o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:€€€€€€€€€ї: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:€€€€€€€€€ї
!
_user_specified_name	input_1
–
У
D__inference_conv1d_1_layer_call_and_return_conditional_losses_188260

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐ"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€В
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ї@Т
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : †
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@Ѓ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ї*
paddingVALID*
strides
Б
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї*
squeeze_dims

э€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0В
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€їU
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€їf
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€їД
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€ї@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€ї@
 
_user_specified_nameinputs
Ё
Ъ
)__inference_conv1d_2_layer_call_fn_190097

inputs
unknown:@
	unknown_0:@
identityИҐStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ї@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_188337t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ї@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€ї: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€ї
 
_user_specified_nameinputs
√
Х
(__inference_dense_1_layer_call_fn_190232

inputs
unknown:@@
	unknown_0:@
identityИҐStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_188409o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
фM
В
A__inference_model_layer_call_and_return_conditional_losses_189247
input_11
multi_head_attention_189169:-
multi_head_attention_189171:1
multi_head_attention_189173:-
multi_head_attention_189175:1
multi_head_attention_189177:-
multi_head_attention_189179:1
multi_head_attention_189181:)
multi_head_attention_189183:#
conv1d_189187:@
conv1d_189189:@%
conv1d_1_189193:@
conv1d_1_189195:3
multi_head_attention_1_189199:/
multi_head_attention_1_189201:3
multi_head_attention_1_189203:/
multi_head_attention_1_189205:3
multi_head_attention_1_189207:/
multi_head_attention_1_189209:3
multi_head_attention_1_189211:+
multi_head_attention_1_189213:%
conv1d_2_189217:@
conv1d_2_189219:@%
conv1d_3_189223:@
conv1d_3_189225:
dense_189231:	ї@
dense_189233:@ 
dense_1_189236:@@
dense_1_189238:@ 
dense_2_189241:@
dense_2_189243:
identityИҐconv1d/StatefulPartitionedCallҐ conv1d_1/StatefulPartitionedCallҐ conv1d_2/StatefulPartitionedCallҐ conv1d_3/StatefulPartitionedCallҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐdense_2/StatefulPartitionedCallҐdropout/StatefulPartitionedCallҐ!dropout_1/StatefulPartitionedCallҐ!dropout_2/StatefulPartitionedCallҐ,multi_head_attention/StatefulPartitionedCallҐ.multi_head_attention_1/StatefulPartitionedCallн
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCallinput_1input_1multi_head_attention_189169multi_head_attention_189171multi_head_attention_189173multi_head_attention_189175multi_head_attention_189177multi_head_attention_189179multi_head_attention_189181multi_head_attention_189183*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ї**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_188790Ъ
tf.__operators__.add/AddV2AddV25multi_head_attention/StatefulPartitionedCall:output:0input_1*
T0*,
_output_shapes
:€€€€€€€€€їИ
conv1d/StatefulPartitionedCallStatefulPartitionedCalltf.__operators__.add/AddV2:z:0conv1d_189187conv1d_189189*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ї@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_188231п
dropout/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ї@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_188709Ъ
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0conv1d_1_189193conv1d_1_189195*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ї*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_188260І
tf.__operators__.add_1/AddV2AddV2)conv1d_1/StatefulPartitionedCall:output:0tf.__operators__.add/AddV2:z:0*
T0*,
_output_shapes
:€€€€€€€€€ї≥
.multi_head_attention_1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_1/AddV2:z:0 tf.__operators__.add_1/AddV2:z:0multi_head_attention_1_189199multi_head_attention_1_189201multi_head_attention_1_189203multi_head_attention_1_189205multi_head_attention_1_189207multi_head_attention_1_189209multi_head_attention_1_189211multi_head_attention_1_189213*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ї**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_188660Ј
tf.__operators__.add_2/AddV2AddV27multi_head_attention_1/StatefulPartitionedCall:output:0 tf.__operators__.add_1/AddV2:z:0*
T0*,
_output_shapes
:€€€€€€€€€їТ
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_2/AddV2:z:0conv1d_2_189217conv1d_2_189219*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ї@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_188337Ч
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ї@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_188579Ь
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0conv1d_3_189223conv1d_3_189225*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ї*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv1d_3_layer_call_and_return_conditional_losses_188366©
tf.__operators__.add_3/AddV2AddV2)conv1d_3/StatefulPartitionedCall:output:0 tf.__operators__.add_2/AddV2:z:0*
T0*,
_output_shapes
:€€€€€€€€€їо
$global_max_pooling1d/PartitionedCallPartitionedCall tf.__operators__.add_3/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ї* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_188151Щ
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall-global_max_pooling1d/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ї* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_188546Л
dense/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_189231dense_189233*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_188392П
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_189236dense_1_189238*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_188409С
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_189241dense_2_189243*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_188426w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ю
NoOpNoOp^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCall/^multi_head_attention_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:€€€€€€€€€ї: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2\
,multi_head_attention/StatefulPartitionedCall,multi_head_attention/StatefulPartitionedCall2`
.multi_head_attention_1/StatefulPartitionedCall.multi_head_attention_1/StatefulPartitionedCall:U Q
,
_output_shapes
:€€€€€€€€€ї
!
_user_specified_name	input_1
И
c
*__inference_dropout_1_layer_call_fn_190123

inputs
identityИҐStatefulPartitionedCall»
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ї@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_188579t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ї@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ї@22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€ї@
 
_user_specified_nameinputs
ђ

d
E__inference_dropout_1_layer_call_and_return_conditional_losses_188579

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Ґ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї@*
dtype0*
seed2€€€€[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=Ђ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї@t
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:€€€€€€€€€ї@n
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:€€€€€€€€€ї@^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:€€€€€€€€€ї@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ї@:T P
,
_output_shapes
:€€€€€€€€€ї@
 
_user_specified_nameinputs
м
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_188348

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:€€€€€€€€€ї@`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ї@:T P
,
_output_shapes
:€€€€€€€€€ї@
 
_user_specified_nameinputs
Я

ф
C__inference_dense_2_layer_call_and_return_conditional_losses_188426

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
яƒ
д/
__inference__traced_save_190583
file_prefix,
(savev2_conv1d_kernel_read_readvariableop*
&savev2_conv1d_bias_read_readvariableop.
*savev2_conv1d_1_kernel_read_readvariableop,
(savev2_conv1d_1_bias_read_readvariableop.
*savev2_conv1d_2_kernel_read_readvariableop,
(savev2_conv1d_2_bias_read_readvariableop.
*savev2_conv1d_3_kernel_read_readvariableop,
(savev2_conv1d_3_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop@
<savev2_multi_head_attention_query_kernel_read_readvariableop>
:savev2_multi_head_attention_query_bias_read_readvariableop>
:savev2_multi_head_attention_key_kernel_read_readvariableop<
8savev2_multi_head_attention_key_bias_read_readvariableop@
<savev2_multi_head_attention_value_kernel_read_readvariableop>
:savev2_multi_head_attention_value_bias_read_readvariableopK
Gsavev2_multi_head_attention_attention_output_kernel_read_readvariableopI
Esavev2_multi_head_attention_attention_output_bias_read_readvariableopB
>savev2_multi_head_attention_1_query_kernel_read_readvariableop@
<savev2_multi_head_attention_1_query_bias_read_readvariableop@
<savev2_multi_head_attention_1_key_kernel_read_readvariableop>
:savev2_multi_head_attention_1_key_bias_read_readvariableopB
>savev2_multi_head_attention_1_value_kernel_read_readvariableop@
<savev2_multi_head_attention_1_value_bias_read_readvariableopM
Isavev2_multi_head_attention_1_attention_output_kernel_read_readvariableopK
Gsavev2_multi_head_attention_1_attention_output_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop3
/savev2_adam_conv1d_kernel_m_read_readvariableop1
-savev2_adam_conv1d_bias_m_read_readvariableop5
1savev2_adam_conv1d_1_kernel_m_read_readvariableop3
/savev2_adam_conv1d_1_bias_m_read_readvariableop5
1savev2_adam_conv1d_2_kernel_m_read_readvariableop3
/savev2_adam_conv1d_2_bias_m_read_readvariableop5
1savev2_adam_conv1d_3_kernel_m_read_readvariableop3
/savev2_adam_conv1d_3_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableopG
Csavev2_adam_multi_head_attention_query_kernel_m_read_readvariableopE
Asavev2_adam_multi_head_attention_query_bias_m_read_readvariableopE
Asavev2_adam_multi_head_attention_key_kernel_m_read_readvariableopC
?savev2_adam_multi_head_attention_key_bias_m_read_readvariableopG
Csavev2_adam_multi_head_attention_value_kernel_m_read_readvariableopE
Asavev2_adam_multi_head_attention_value_bias_m_read_readvariableopR
Nsavev2_adam_multi_head_attention_attention_output_kernel_m_read_readvariableopP
Lsavev2_adam_multi_head_attention_attention_output_bias_m_read_readvariableopI
Esavev2_adam_multi_head_attention_1_query_kernel_m_read_readvariableopG
Csavev2_adam_multi_head_attention_1_query_bias_m_read_readvariableopG
Csavev2_adam_multi_head_attention_1_key_kernel_m_read_readvariableopE
Asavev2_adam_multi_head_attention_1_key_bias_m_read_readvariableopI
Esavev2_adam_multi_head_attention_1_value_kernel_m_read_readvariableopG
Csavev2_adam_multi_head_attention_1_value_bias_m_read_readvariableopT
Psavev2_adam_multi_head_attention_1_attention_output_kernel_m_read_readvariableopR
Nsavev2_adam_multi_head_attention_1_attention_output_bias_m_read_readvariableop3
/savev2_adam_conv1d_kernel_v_read_readvariableop1
-savev2_adam_conv1d_bias_v_read_readvariableop5
1savev2_adam_conv1d_1_kernel_v_read_readvariableop3
/savev2_adam_conv1d_1_bias_v_read_readvariableop5
1savev2_adam_conv1d_2_kernel_v_read_readvariableop3
/savev2_adam_conv1d_2_bias_v_read_readvariableop5
1savev2_adam_conv1d_3_kernel_v_read_readvariableop3
/savev2_adam_conv1d_3_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableopG
Csavev2_adam_multi_head_attention_query_kernel_v_read_readvariableopE
Asavev2_adam_multi_head_attention_query_bias_v_read_readvariableopE
Asavev2_adam_multi_head_attention_key_kernel_v_read_readvariableopC
?savev2_adam_multi_head_attention_key_bias_v_read_readvariableopG
Csavev2_adam_multi_head_attention_value_kernel_v_read_readvariableopE
Asavev2_adam_multi_head_attention_value_bias_v_read_readvariableopR
Nsavev2_adam_multi_head_attention_attention_output_kernel_v_read_readvariableopP
Lsavev2_adam_multi_head_attention_attention_output_bias_v_read_readvariableopI
Esavev2_adam_multi_head_attention_1_query_kernel_v_read_readvariableopG
Csavev2_adam_multi_head_attention_1_query_bias_v_read_readvariableopG
Csavev2_adam_multi_head_attention_1_key_kernel_v_read_readvariableopE
Asavev2_adam_multi_head_attention_1_key_bias_v_read_readvariableopI
Esavev2_adam_multi_head_attention_1_value_kernel_v_read_readvariableopG
Csavev2_adam_multi_head_attention_1_value_bias_v_read_readvariableopT
Psavev2_adam_multi_head_attention_1_attention_output_kernel_v_read_readvariableopR
Nsavev2_adam_multi_head_attention_1_attention_output_bias_v_read_readvariableop
savev2_const

identity_1ИҐMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: х2
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:d*
dtype0*Ю2
valueФ2BС2dB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЄ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:d*
dtype0*Ё
value”B–dB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B €-
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv1d_kernel_read_readvariableop&savev2_conv1d_bias_read_readvariableop*savev2_conv1d_1_kernel_read_readvariableop(savev2_conv1d_1_bias_read_readvariableop*savev2_conv1d_2_kernel_read_readvariableop(savev2_conv1d_2_bias_read_readvariableop*savev2_conv1d_3_kernel_read_readvariableop(savev2_conv1d_3_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop<savev2_multi_head_attention_query_kernel_read_readvariableop:savev2_multi_head_attention_query_bias_read_readvariableop:savev2_multi_head_attention_key_kernel_read_readvariableop8savev2_multi_head_attention_key_bias_read_readvariableop<savev2_multi_head_attention_value_kernel_read_readvariableop:savev2_multi_head_attention_value_bias_read_readvariableopGsavev2_multi_head_attention_attention_output_kernel_read_readvariableopEsavev2_multi_head_attention_attention_output_bias_read_readvariableop>savev2_multi_head_attention_1_query_kernel_read_readvariableop<savev2_multi_head_attention_1_query_bias_read_readvariableop<savev2_multi_head_attention_1_key_kernel_read_readvariableop:savev2_multi_head_attention_1_key_bias_read_readvariableop>savev2_multi_head_attention_1_value_kernel_read_readvariableop<savev2_multi_head_attention_1_value_bias_read_readvariableopIsavev2_multi_head_attention_1_attention_output_kernel_read_readvariableopGsavev2_multi_head_attention_1_attention_output_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop/savev2_adam_conv1d_kernel_m_read_readvariableop-savev2_adam_conv1d_bias_m_read_readvariableop1savev2_adam_conv1d_1_kernel_m_read_readvariableop/savev2_adam_conv1d_1_bias_m_read_readvariableop1savev2_adam_conv1d_2_kernel_m_read_readvariableop/savev2_adam_conv1d_2_bias_m_read_readvariableop1savev2_adam_conv1d_3_kernel_m_read_readvariableop/savev2_adam_conv1d_3_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableopCsavev2_adam_multi_head_attention_query_kernel_m_read_readvariableopAsavev2_adam_multi_head_attention_query_bias_m_read_readvariableopAsavev2_adam_multi_head_attention_key_kernel_m_read_readvariableop?savev2_adam_multi_head_attention_key_bias_m_read_readvariableopCsavev2_adam_multi_head_attention_value_kernel_m_read_readvariableopAsavev2_adam_multi_head_attention_value_bias_m_read_readvariableopNsavev2_adam_multi_head_attention_attention_output_kernel_m_read_readvariableopLsavev2_adam_multi_head_attention_attention_output_bias_m_read_readvariableopEsavev2_adam_multi_head_attention_1_query_kernel_m_read_readvariableopCsavev2_adam_multi_head_attention_1_query_bias_m_read_readvariableopCsavev2_adam_multi_head_attention_1_key_kernel_m_read_readvariableopAsavev2_adam_multi_head_attention_1_key_bias_m_read_readvariableopEsavev2_adam_multi_head_attention_1_value_kernel_m_read_readvariableopCsavev2_adam_multi_head_attention_1_value_bias_m_read_readvariableopPsavev2_adam_multi_head_attention_1_attention_output_kernel_m_read_readvariableopNsavev2_adam_multi_head_attention_1_attention_output_bias_m_read_readvariableop/savev2_adam_conv1d_kernel_v_read_readvariableop-savev2_adam_conv1d_bias_v_read_readvariableop1savev2_adam_conv1d_1_kernel_v_read_readvariableop/savev2_adam_conv1d_1_bias_v_read_readvariableop1savev2_adam_conv1d_2_kernel_v_read_readvariableop/savev2_adam_conv1d_2_bias_v_read_readvariableop1savev2_adam_conv1d_3_kernel_v_read_readvariableop/savev2_adam_conv1d_3_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableopCsavev2_adam_multi_head_attention_query_kernel_v_read_readvariableopAsavev2_adam_multi_head_attention_query_bias_v_read_readvariableopAsavev2_adam_multi_head_attention_key_kernel_v_read_readvariableop?savev2_adam_multi_head_attention_key_bias_v_read_readvariableopCsavev2_adam_multi_head_attention_value_kernel_v_read_readvariableopAsavev2_adam_multi_head_attention_value_bias_v_read_readvariableopNsavev2_adam_multi_head_attention_attention_output_kernel_v_read_readvariableopLsavev2_adam_multi_head_attention_attention_output_bias_v_read_readvariableopEsavev2_adam_multi_head_attention_1_query_kernel_v_read_readvariableopCsavev2_adam_multi_head_attention_1_query_bias_v_read_readvariableopCsavev2_adam_multi_head_attention_1_key_kernel_v_read_readvariableopAsavev2_adam_multi_head_attention_1_key_bias_v_read_readvariableopEsavev2_adam_multi_head_attention_1_value_kernel_v_read_readvariableopCsavev2_adam_multi_head_attention_1_value_bias_v_read_readvariableopPsavev2_adam_multi_head_attention_1_attention_output_kernel_v_read_readvariableopNsavev2_adam_multi_head_attention_1_attention_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *r
dtypesh
f2d	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*÷
_input_shapesƒ
Ѕ: :@:@:@::@:@:@::	ї@:@:@@:@:@:: : : : : ::::::::::::::::: : : : :@:@:@::@:@:@::	ї@:@:@@:@:@::::::::::::::::::@:@:@::@:@:@::	ї@:@:@@:@:@:::::::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
:@: 

_output_shapes
:@:($
"
_output_shapes
:@: 

_output_shapes
::($
"
_output_shapes
:@: 

_output_shapes
:@:($
"
_output_shapes
:@: 

_output_shapes
::%	!

_output_shapes
:	ї@: 


_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :($
"
_output_shapes
::$ 

_output_shapes

::($
"
_output_shapes
::$ 

_output_shapes

::($
"
_output_shapes
::$ 

_output_shapes

::($
"
_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
::$ 

_output_shapes

::($
"
_output_shapes
::$ 

_output_shapes

::( $
"
_output_shapes
::$! 

_output_shapes

::("$
"
_output_shapes
:: #

_output_shapes
::$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(($
"
_output_shapes
:@: )

_output_shapes
:@:(*$
"
_output_shapes
:@: +

_output_shapes
::(,$
"
_output_shapes
:@: -

_output_shapes
:@:(.$
"
_output_shapes
:@: /

_output_shapes
::%0!

_output_shapes
:	ї@: 1

_output_shapes
:@:$2 

_output_shapes

:@@: 3

_output_shapes
:@:$4 

_output_shapes

:@: 5

_output_shapes
::(6$
"
_output_shapes
::$7 

_output_shapes

::(8$
"
_output_shapes
::$9 

_output_shapes

::(:$
"
_output_shapes
::$; 

_output_shapes

::(<$
"
_output_shapes
:: =

_output_shapes
::(>$
"
_output_shapes
::$? 

_output_shapes

::(@$
"
_output_shapes
::$A 

_output_shapes

::(B$
"
_output_shapes
::$C 

_output_shapes

::(D$
"
_output_shapes
:: E

_output_shapes
::(F$
"
_output_shapes
:@: G

_output_shapes
:@:(H$
"
_output_shapes
:@: I

_output_shapes
::(J$
"
_output_shapes
:@: K

_output_shapes
:@:(L$
"
_output_shapes
:@: M

_output_shapes
::%N!

_output_shapes
:	ї@: O

_output_shapes
:@:$P 

_output_shapes

:@@: Q

_output_shapes
:@:$R 

_output_shapes

:@: S

_output_shapes
::(T$
"
_output_shapes
::$U 

_output_shapes

::(V$
"
_output_shapes
::$W 

_output_shapes

::(X$
"
_output_shapes
::$Y 

_output_shapes

::(Z$
"
_output_shapes
:: [

_output_shapes
::(\$
"
_output_shapes
::$] 

_output_shapes

::(^$
"
_output_shapes
::$_ 

_output_shapes

::(`$
"
_output_shapes
::$a 

_output_shapes

::(b$
"
_output_shapes
:: c

_output_shapes
::d

_output_shapes
: 
Ё
Ъ
)__inference_conv1d_3_layer_call_fn_190149

inputs
unknown:@
	unknown_0:
identityИҐStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ї*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv1d_3_layer_call_and_return_conditional_losses_188366t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ї`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€ї@: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€ї@
 
_user_specified_nameinputs
–
У
D__inference_conv1d_2_layer_call_and_return_conditional_losses_190113

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐ"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€В
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€їТ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : †
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@Ѓ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ї@*
paddingVALID*
strides
Б
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї@*
squeeze_dims

э€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0В
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€ї@U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї@f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ї@Д
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€ї: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€ї
 
_user_specified_nameinputs
¶
F
*__inference_dropout_2_layer_call_fn_190181

inputs
identityі
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ї* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_188379a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€ї"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€ї:P L
(
_output_shapes
:€€€€€€€€€ї
 
_user_specified_nameinputs
ґ
F
*__inference_dropout_1_layer_call_fn_190118

inputs
identityЄ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ї@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_188348e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ї@:T P
,
_output_shapes
:€€€€€€€€€ї@
 
_user_specified_nameinputs
У2
ы
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_190088	
query	
valueA
+query_einsum_einsum_readvariableop_resource:3
!query_add_readvariableop_resource:?
)key_einsum_einsum_readvariableop_resource:1
key_add_readvariableop_resource:A
+value_einsum_einsum_readvariableop_resource:3
!value_add_readvariableop_resource:L
6attention_output_einsum_einsum_readvariableop_resource::
,attention_output_add_readvariableop_resource:
identityИҐ#attention_output/add/ReadVariableOpҐ-attention_output/einsum/Einsum/ReadVariableOpҐkey/add/ReadVariableOpҐ key/einsum/Einsum/ReadVariableOpҐquery/add/ReadVariableOpҐ"query/einsum/Einsum/ReadVariableOpҐvalue/add/ReadVariableOpҐ"value/einsum/Einsum/ReadVariableOpТ
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0±
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:*
dtype0Н
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€їО
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0≠
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:*
dtype0З
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€їТ
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0±
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:*
dtype0Н
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€їJ
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *уµ>d
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€їС
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:€€€€€€€€€її*
equationaecd,abcd->acben
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€їїZ
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?С
dropout/dropout/MulMulsoftmax/Softmax:softmax:0dropout/dropout/Const:output:0*
T0*1
_output_shapes
:€€€€€€€€€її^
dropout/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
:Ј
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*1
_output_shapes
:€€€€€€€€€її*
dtype0*
seed2€€€€c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=»
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:€€€€€€€€€їїЙ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:€€€€€€€€€їїЛ
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:€€€€€€€€€її¶
einsum_1/EinsumEinsumdropout/dropout/Mul_1:z:0value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationacbe,aecd->abcd®
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0÷
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€ї*
equationabcd,cde->abeМ
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0™
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€їl
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€їЎ
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:€€€€€€€€€ї:€€€€€€€€€ї: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:€€€€€€€€€ї

_user_specified_namequery:SO
,
_output_shapes
:€€€€€€€€€ї

_user_specified_namevalue
К
м
7__inference_multi_head_attention_1_layer_call_fn_189989	
query	
value
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identityИҐStatefulPartitionedCallƒ
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ї**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_188302t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ї`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:€€€€€€€€€ї:€€€€€€€€€ї: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
,
_output_shapes
:€€€€€€€€€ї

_user_specified_namequery:SO
,
_output_shapes
:€€€€€€€€€ї

_user_specified_namevalue
–
У
D__inference_conv1d_3_layer_call_and_return_conditional_losses_188366

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐ"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€В
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ї@Т
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : †
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@Ѓ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ї*
paddingVALID*
strides
Б
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї*
squeeze_dims

э€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0В
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€їU
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€їf
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€їД
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€ї@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€ї@
 
_user_specified_nameinputs
С2
щ
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_188790	
query	
valueA
+query_einsum_einsum_readvariableop_resource:3
!query_add_readvariableop_resource:?
)key_einsum_einsum_readvariableop_resource:1
key_add_readvariableop_resource:A
+value_einsum_einsum_readvariableop_resource:3
!value_add_readvariableop_resource:L
6attention_output_einsum_einsum_readvariableop_resource::
,attention_output_add_readvariableop_resource:
identityИҐ#attention_output/add/ReadVariableOpҐ-attention_output/einsum/Einsum/ReadVariableOpҐkey/add/ReadVariableOpҐ key/einsum/Einsum/ReadVariableOpҐquery/add/ReadVariableOpҐ"query/einsum/Einsum/ReadVariableOpҐvalue/add/ReadVariableOpҐ"value/einsum/Einsum/ReadVariableOpТ
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0±
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:*
dtype0Н
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€їО
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0≠
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:*
dtype0З
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€їТ
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0±
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:*
dtype0Н
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€їJ
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *уµ>d
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€їС
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:€€€€€€€€€її*
equationaecd,abcd->acben
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€їїZ
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?С
dropout/dropout/MulMulsoftmax/Softmax:softmax:0dropout/dropout/Const:output:0*
T0*1
_output_shapes
:€€€€€€€€€її^
dropout/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
:Ј
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*1
_output_shapes
:€€€€€€€€€її*
dtype0*
seed2€€€€c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=»
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:€€€€€€€€€їїЙ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:€€€€€€€€€їїЛ
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:€€€€€€€€€її¶
einsum_1/EinsumEinsumdropout/dropout/Mul_1:z:0value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationacbe,aecd->abcd®
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0÷
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€ї*
equationabcd,cde->abeМ
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0™
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€їl
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€їЎ
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:€€€€€€€€€ї:€€€€€€€€€ї: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:€€€€€€€€€ї

_user_specified_namequery:SO
,
_output_shapes
:€€€€€€€€€ї

_user_specified_namevalue
–
У
D__inference_conv1d_2_layer_call_and_return_conditional_losses_188337

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐ"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€В
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€їТ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : †
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@Ѓ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ї@*
paddingVALID*
strides
Б
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї@*
squeeze_dims

э€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0В
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€ї@U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї@f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ї@Д
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€ї: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€ї
 
_user_specified_nameinputs
№
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_188379

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:€€€€€€€€€ї\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:€€€€€€€€€ї"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€ї:P L
(
_output_shapes
:€€€€€€€€€ї
 
_user_specified_nameinputs
к
a
C__inference_dropout_layer_call_and_return_conditional_losses_189930

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:€€€€€€€€€ї@`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ї@:T P
,
_output_shapes
:€€€€€€€€€ї@
 
_user_specified_nameinputs
М

d
E__inference_dropout_2_layer_call_and_return_conditional_losses_188546

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€їC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Ю
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ї*
dtype0*
seed2€€€€[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>І
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€їp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€їj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€їZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€ї"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€ї:P L
(
_output_shapes
:€€€€€€€€€ї
 
_user_specified_nameinputs
™

b
C__inference_dropout_layer_call_and_return_conditional_losses_189942

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Ґ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї@*
dtype0*
seed2€€€€[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=Ђ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї@t
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:€€€€€€€€€ї@n
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:€€€€€€€€€ї@^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:€€€€€€€€€ї@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ї@:T P
,
_output_shapes
:€€€€€€€€€ї@
 
_user_specified_nameinputs
Ъ

ф
C__inference_dense_1_layer_call_and_return_conditional_losses_188409

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
пM
Б
A__inference_model_layer_call_and_return_conditional_losses_188957

inputs1
multi_head_attention_188879:-
multi_head_attention_188881:1
multi_head_attention_188883:-
multi_head_attention_188885:1
multi_head_attention_188887:-
multi_head_attention_188889:1
multi_head_attention_188891:)
multi_head_attention_188893:#
conv1d_188897:@
conv1d_188899:@%
conv1d_1_188903:@
conv1d_1_188905:3
multi_head_attention_1_188909:/
multi_head_attention_1_188911:3
multi_head_attention_1_188913:/
multi_head_attention_1_188915:3
multi_head_attention_1_188917:/
multi_head_attention_1_188919:3
multi_head_attention_1_188921:+
multi_head_attention_1_188923:%
conv1d_2_188927:@
conv1d_2_188929:@%
conv1d_3_188933:@
conv1d_3_188935:
dense_188941:	ї@
dense_188943:@ 
dense_1_188946:@@
dense_1_188948:@ 
dense_2_188951:@
dense_2_188953:
identityИҐconv1d/StatefulPartitionedCallҐ conv1d_1/StatefulPartitionedCallҐ conv1d_2/StatefulPartitionedCallҐ conv1d_3/StatefulPartitionedCallҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐdense_2/StatefulPartitionedCallҐdropout/StatefulPartitionedCallҐ!dropout_1/StatefulPartitionedCallҐ!dropout_2/StatefulPartitionedCallҐ,multi_head_attention/StatefulPartitionedCallҐ.multi_head_attention_1/StatefulPartitionedCallл
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCallinputsinputsmulti_head_attention_188879multi_head_attention_188881multi_head_attention_188883multi_head_attention_188885multi_head_attention_188887multi_head_attention_188889multi_head_attention_188891multi_head_attention_188893*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ї**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_188790Щ
tf.__operators__.add/AddV2AddV25multi_head_attention/StatefulPartitionedCall:output:0inputs*
T0*,
_output_shapes
:€€€€€€€€€їИ
conv1d/StatefulPartitionedCallStatefulPartitionedCalltf.__operators__.add/AddV2:z:0conv1d_188897conv1d_188899*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ї@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_188231п
dropout/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ї@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_188709Ъ
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0conv1d_1_188903conv1d_1_188905*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ї*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_188260І
tf.__operators__.add_1/AddV2AddV2)conv1d_1/StatefulPartitionedCall:output:0tf.__operators__.add/AddV2:z:0*
T0*,
_output_shapes
:€€€€€€€€€ї≥
.multi_head_attention_1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_1/AddV2:z:0 tf.__operators__.add_1/AddV2:z:0multi_head_attention_1_188909multi_head_attention_1_188911multi_head_attention_1_188913multi_head_attention_1_188915multi_head_attention_1_188917multi_head_attention_1_188919multi_head_attention_1_188921multi_head_attention_1_188923*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ї**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_188660Ј
tf.__operators__.add_2/AddV2AddV27multi_head_attention_1/StatefulPartitionedCall:output:0 tf.__operators__.add_1/AddV2:z:0*
T0*,
_output_shapes
:€€€€€€€€€їТ
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_2/AddV2:z:0conv1d_2_188927conv1d_2_188929*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ї@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_188337Ч
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ї@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_188579Ь
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0conv1d_3_188933conv1d_3_188935*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ї*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv1d_3_layer_call_and_return_conditional_losses_188366©
tf.__operators__.add_3/AddV2AddV2)conv1d_3/StatefulPartitionedCall:output:0 tf.__operators__.add_2/AddV2:z:0*
T0*,
_output_shapes
:€€€€€€€€€їо
$global_max_pooling1d/PartitionedCallPartitionedCall tf.__operators__.add_3/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ї* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_188151Щ
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall-global_max_pooling1d/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ї* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_188546Л
dense/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_188941dense_188943*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_188392П
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_188946dense_1_188948*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_188409С
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_188951dense_2_188953*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_188426w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ю
NoOpNoOp^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCall/^multi_head_attention_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:€€€€€€€€€ї: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2\
,multi_head_attention/StatefulPartitionedCall,multi_head_attention/StatefulPartitionedCall2`
.multi_head_attention_1/StatefulPartitionedCall.multi_head_attention_1/StatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€ї
 
_user_specified_nameinputs
ќ
С
B__inference_conv1d_layer_call_and_return_conditional_losses_189915

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐ"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€В
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€їТ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : †
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@Ѓ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ї@*
paddingVALID*
strides
Б
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї@*
squeeze_dims

э€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0В
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€ї@U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї@f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ї@Д
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€ї: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€ї
 
_user_specified_nameinputs
ќ
С
B__inference_conv1d_layer_call_and_return_conditional_losses_188231

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐ"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€В
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€їТ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : †
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@Ѓ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ї@*
paddingVALID*
strides
Б
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї@*
squeeze_dims

э€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0В
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€ї@U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї@f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ї@Д
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€ї: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€ї
 
_user_specified_nameinputs
–
г
&__inference_model_layer_call_fn_189318

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10: 

unknown_11:

unknown_12: 

unknown_13:

unknown_14: 

unknown_15:

unknown_16: 

unknown_17:

unknown_18: 

unknown_19:@

unknown_20:@ 

unknown_21:@

unknown_22:

unknown_23:	ї@

unknown_24:@

unknown_25:@@

unknown_26:@

unknown_27:@

unknown_28:
identityИҐStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*@
_read_only_resource_inputs"
 	
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_188433o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:€€€€€€€€€ї: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€ї
 
_user_specified_nameinputs
ш
c
*__inference_dropout_2_layer_call_fn_190186

inputs
identityИҐStatefulPartitionedCallƒ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ї* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_188546p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€ї`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€ї22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€ї
 
_user_specified_nameinputs
к
a
C__inference_dropout_layer_call_and_return_conditional_losses_188242

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:€€€€€€€€€ї@`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ї@:T P
,
_output_shapes
:€€€€€€€€€ї@
 
_user_specified_nameinputs
У2
ы
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_188660	
query	
valueA
+query_einsum_einsum_readvariableop_resource:3
!query_add_readvariableop_resource:?
)key_einsum_einsum_readvariableop_resource:1
key_add_readvariableop_resource:A
+value_einsum_einsum_readvariableop_resource:3
!value_add_readvariableop_resource:L
6attention_output_einsum_einsum_readvariableop_resource::
,attention_output_add_readvariableop_resource:
identityИҐ#attention_output/add/ReadVariableOpҐ-attention_output/einsum/Einsum/ReadVariableOpҐkey/add/ReadVariableOpҐ key/einsum/Einsum/ReadVariableOpҐquery/add/ReadVariableOpҐ"query/einsum/Einsum/ReadVariableOpҐvalue/add/ReadVariableOpҐ"value/einsum/Einsum/ReadVariableOpТ
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0±
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:*
dtype0Н
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€їО
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0≠
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:*
dtype0З
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€їТ
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0±
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:*
dtype0Н
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€їJ
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *уµ>d
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€їС
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:€€€€€€€€€її*
equationaecd,abcd->acben
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€їїZ
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?С
dropout/dropout/MulMulsoftmax/Softmax:softmax:0dropout/dropout/Const:output:0*
T0*1
_output_shapes
:€€€€€€€€€її^
dropout/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
:Ј
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*1
_output_shapes
:€€€€€€€€€її*
dtype0*
seed2€€€€c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=»
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:€€€€€€€€€їїЙ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:€€€€€€€€€їїЛ
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:€€€€€€€€€її¶
einsum_1/EinsumEinsumdropout/dropout/Mul_1:z:0value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationacbe,aecd->abcd®
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0÷
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€ї*
equationabcd,cde->abeМ
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0™
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€їl
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€їЎ
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:€€€€€€€€€ї:€€€€€€€€€ї: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:€€€€€€€€€ї

_user_specified_namequery:SO
,
_output_shapes
:€€€€€€€€€ї

_user_specified_namevalue
”
д
&__inference_model_layer_call_fn_189085
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10: 

unknown_11:

unknown_12: 

unknown_13:

unknown_14: 

unknown_15:

unknown_16: 

unknown_17:

unknown_18: 

unknown_19:@

unknown_20:@ 

unknown_21:@

unknown_22:

unknown_23:	ї@

unknown_24:@

unknown_25:@@

unknown_26:@

unknown_27:@

unknown_28:
identityИҐStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*@
_read_only_resource_inputs"
 	
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_188957o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:€€€€€€€€€ї: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:€€€€€€€€€ї
!
_user_specified_name	input_1
Ж
к
5__inference_multi_head_attention_layer_call_fn_189791	
query	
value
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identityИҐStatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ї**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_188196t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ї`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:€€€€€€€€€ї:€€€€€€€€€ї: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
,
_output_shapes
:€€€€€€€€€ї

_user_specified_namequery:SO
,
_output_shapes
:€€€€€€€€€ї

_user_specified_namevalue
м
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_190128

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:€€€€€€€€€ї@`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ї@:T P
,
_output_shapes
:€€€€€€€€€ї@
 
_user_specified_nameinputs
Ь

у
A__inference_dense_layer_call_and_return_conditional_losses_188392

inputs1
matmul_readvariableop_resource:	ї@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ї@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ї: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€ї
 
_user_specified_nameinputs
М

d
E__inference_dropout_2_layer_call_and_return_conditional_losses_190203

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€їC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Ю
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ї*
dtype0*
seed2€€€€[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>І
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€їp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€їj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€їZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€ї"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€ї:P L
(
_output_shapes
:€€€€€€€€€ї
 
_user_specified_nameinputs
Ё
Ъ
)__inference_conv1d_1_layer_call_fn_189951

inputs
unknown:@
	unknown_0:
identityИҐStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ї*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_188260t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ї`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€ї@: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€ї@
 
_user_specified_nameinputs
і*
ы
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_190046	
query	
valueA
+query_einsum_einsum_readvariableop_resource:3
!query_add_readvariableop_resource:?
)key_einsum_einsum_readvariableop_resource:1
key_add_readvariableop_resource:A
+value_einsum_einsum_readvariableop_resource:3
!value_add_readvariableop_resource:L
6attention_output_einsum_einsum_readvariableop_resource::
,attention_output_add_readvariableop_resource:
identityИҐ#attention_output/add/ReadVariableOpҐ-attention_output/einsum/Einsum/ReadVariableOpҐkey/add/ReadVariableOpҐ key/einsum/Einsum/ReadVariableOpҐquery/add/ReadVariableOpҐ"query/einsum/Einsum/ReadVariableOpҐvalue/add/ReadVariableOpҐ"value/einsum/Einsum/ReadVariableOpТ
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0±
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:*
dtype0Н
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€їО
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0≠
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:*
dtype0З
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€їТ
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0±
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:*
dtype0Н
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€їJ
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *уµ>d
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€їС
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:€€€€€€€€€її*
equationaecd,abcd->acben
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€їїs
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*1
_output_shapes
:€€€€€€€€€її¶
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationacbe,aecd->abcd®
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0÷
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€ї*
equationabcd,cde->abeМ
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0™
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€їl
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€їЎ
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:€€€€€€€€€ї:€€€€€€€€€ї: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:€€€€€€€€€ї

_user_specified_namequery:SO
,
_output_shapes
:€€€€€€€€€ї

_user_specified_namevalue
ў
Ш
'__inference_conv1d_layer_call_fn_189899

inputs
unknown:@
	unknown_0:@
identityИҐStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ї@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_188231t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ї@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€ї: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€ї
 
_user_specified_nameinputs
™

b
C__inference_dropout_layer_call_and_return_conditional_losses_188709

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Ґ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї@*
dtype0*
seed2€€€€[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=Ђ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї@t
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:€€€€€€€€€ї@n
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:€€€€€€€€€ї@^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:€€€€€€€€€ї@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ї@:T P
,
_output_shapes
:€€€€€€€€€ї@
 
_user_specified_nameinputs
Ъ

ф
C__inference_dense_1_layer_call_and_return_conditional_losses_190243

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Г
l
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_190176

inputs
identityW
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :m
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€]
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ь

у
A__inference_dense_layer_call_and_return_conditional_losses_190223

inputs1
matmul_readvariableop_resource:	ї@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ї@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ї: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€ї
 
_user_specified_nameinputs
ђ

d
E__inference_dropout_1_layer_call_and_return_conditional_losses_190140

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Ґ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї@*
dtype0*
seed2€€€€[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=Ђ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї@t
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:€€€€€€€€€ї@n
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:€€€€€€€€€ї@^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:€€€€€€€€€ї@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ї@:T P
,
_output_shapes
:€€€€€€€€€ї@
 
_user_specified_nameinputs
ОЌ
ч
A__inference_model_layer_call_and_return_conditional_losses_189525

inputsV
@multi_head_attention_query_einsum_einsum_readvariableop_resource:H
6multi_head_attention_query_add_readvariableop_resource:T
>multi_head_attention_key_einsum_einsum_readvariableop_resource:F
4multi_head_attention_key_add_readvariableop_resource:V
@multi_head_attention_value_einsum_einsum_readvariableop_resource:H
6multi_head_attention_value_add_readvariableop_resource:a
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource:O
Amulti_head_attention_attention_output_add_readvariableop_resource:H
2conv1d_conv1d_expanddims_1_readvariableop_resource:@4
&conv1d_biasadd_readvariableop_resource:@J
4conv1d_1_conv1d_expanddims_1_readvariableop_resource:@6
(conv1d_1_biasadd_readvariableop_resource:X
Bmulti_head_attention_1_query_einsum_einsum_readvariableop_resource:J
8multi_head_attention_1_query_add_readvariableop_resource:V
@multi_head_attention_1_key_einsum_einsum_readvariableop_resource:H
6multi_head_attention_1_key_add_readvariableop_resource:X
Bmulti_head_attention_1_value_einsum_einsum_readvariableop_resource:J
8multi_head_attention_1_value_add_readvariableop_resource:c
Mmulti_head_attention_1_attention_output_einsum_einsum_readvariableop_resource:Q
Cmulti_head_attention_1_attention_output_add_readvariableop_resource:J
4conv1d_2_conv1d_expanddims_1_readvariableop_resource:@6
(conv1d_2_biasadd_readvariableop_resource:@J
4conv1d_3_conv1d_expanddims_1_readvariableop_resource:@6
(conv1d_3_biasadd_readvariableop_resource:7
$dense_matmul_readvariableop_resource:	ї@3
%dense_biasadd_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@@5
'dense_1_biasadd_readvariableop_resource:@8
&dense_2_matmul_readvariableop_resource:@5
'dense_2_biasadd_readvariableop_resource:
identityИҐconv1d/BiasAdd/ReadVariableOpҐ)conv1d/Conv1D/ExpandDims_1/ReadVariableOpҐconv1d_1/BiasAdd/ReadVariableOpҐ+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpҐconv1d_2/BiasAdd/ReadVariableOpҐ+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpҐconv1d_3/BiasAdd/ReadVariableOpҐ+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOpҐdense/BiasAdd/ReadVariableOpҐdense/MatMul/ReadVariableOpҐdense_1/BiasAdd/ReadVariableOpҐdense_1/MatMul/ReadVariableOpҐdense_2/BiasAdd/ReadVariableOpҐdense_2/MatMul/ReadVariableOpҐ8multi_head_attention/attention_output/add/ReadVariableOpҐBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpҐ+multi_head_attention/key/add/ReadVariableOpҐ5multi_head_attention/key/einsum/Einsum/ReadVariableOpҐ-multi_head_attention/query/add/ReadVariableOpҐ7multi_head_attention/query/einsum/Einsum/ReadVariableOpҐ-multi_head_attention/value/add/ReadVariableOpҐ7multi_head_attention/value/einsum/Einsum/ReadVariableOpҐ:multi_head_attention_1/attention_output/add/ReadVariableOpҐDmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpҐ-multi_head_attention_1/key/add/ReadVariableOpҐ7multi_head_attention_1/key/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_1/query/add/ReadVariableOpҐ9multi_head_attention_1/query/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_1/value/add/ReadVariableOpҐ9multi_head_attention_1/value/einsum/Einsum/ReadVariableOpЉ
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0№
(multi_head_attention/query/einsum/EinsumEinsuminputs?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationabc,cde->abde§
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0ћ
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€їЄ
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0Ў
&multi_head_attention/key/einsum/EinsumEinsuminputs=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationabc,cde->abde†
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0∆
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€їЉ
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0№
(multi_head_attention/value/einsum/EinsumEinsuminputs?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationabc,cde->abde§
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0ћ
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ї_
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *уµ>£
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€ї–
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*1
_output_shapes
:€€€€€€€€€її*
equationaecd,abcd->acbeШ
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€їїЭ
%multi_head_attention/dropout/IdentityIdentity.multi_head_attention/softmax/Softmax:softmax:0*
T0*1
_output_shapes
:€€€€€€€€€їїе
$multi_head_attention/einsum_1/EinsumEinsum.multi_head_attention/dropout/Identity:output:0"multi_head_attention/value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationacbe,aecd->abcd“
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0Х
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€ї*
equationabcd,cde->abeґ
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0й
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€їС
tf.__operators__.add/AddV2AddV2-multi_head_attention/attention_output/add:z:0inputs*
T0*,
_output_shapes
:€€€€€€€€€їg
conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€®
conv1d/Conv1D/ExpandDims
ExpandDimstf.__operators__.add/AddV2:z:0%conv1d/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ї†
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0`
conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : µ
conv1d/Conv1D/ExpandDims_1
ExpandDims1conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@√
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ї@*
paddingVALID*
strides
П
conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї@*
squeeze_dims

э€€€€€€€€А
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ч
conv1d/BiasAddBiasAddconv1d/Conv1D/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€ї@c
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї@n
dropout/IdentityIdentityconv1d/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€ї@i
conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€І
conv1d_1/Conv1D/ExpandDims
ExpandDimsdropout/Identity:output:0'conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ї@§
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0b
 conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ї
conv1d_1/Conv1D/ExpandDims_1
ExpandDims3conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@…
conv1d_1/Conv1DConv2D#conv1d_1/Conv1D/ExpandDims:output:0%conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ї*
paddingVALID*
strides
У
conv1d_1/Conv1D/SqueezeSqueezeconv1d_1/Conv1D:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї*
squeeze_dims

э€€€€€€€€Д
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Э
conv1d_1/BiasAddBiasAdd conv1d_1/Conv1D/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€їg
conv1d_1/ReluReluconv1d_1/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€їЩ
tf.__operators__.add_1/AddV2AddV2conv1d_1/Relu:activations:0tf.__operators__.add/AddV2:z:0*
T0*,
_output_shapes
:€€€€€€€€€їј
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_1_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0ъ
*multi_head_attention_1/query/einsum/EinsumEinsum tf.__operators__.add_1/AddV2:z:0Amulti_head_attention_1/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationabc,cde->abde®
/multi_head_attention_1/query/add/ReadVariableOpReadVariableOp8multi_head_attention_1_query_add_readvariableop_resource*
_output_shapes

:*
dtype0“
 multi_head_attention_1/query/addAddV23multi_head_attention_1/query/einsum/Einsum:output:07multi_head_attention_1/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€їЉ
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_1_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0ц
(multi_head_attention_1/key/einsum/EinsumEinsum tf.__operators__.add_1/AddV2:z:0?multi_head_attention_1/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationabc,cde->abde§
-multi_head_attention_1/key/add/ReadVariableOpReadVariableOp6multi_head_attention_1_key_add_readvariableop_resource*
_output_shapes

:*
dtype0ћ
multi_head_attention_1/key/addAddV21multi_head_attention_1/key/einsum/Einsum:output:05multi_head_attention_1/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€їј
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_1_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0ъ
*multi_head_attention_1/value/einsum/EinsumEinsum tf.__operators__.add_1/AddV2:z:0Amulti_head_attention_1/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationabc,cde->abde®
/multi_head_attention_1/value/add/ReadVariableOpReadVariableOp8multi_head_attention_1_value_add_readvariableop_resource*
_output_shapes

:*
dtype0“
 multi_head_attention_1/value/addAddV23multi_head_attention_1/value/einsum/Einsum:output:07multi_head_attention_1/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€їa
multi_head_attention_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *уµ>©
multi_head_attention_1/MulMul$multi_head_attention_1/query/add:z:0%multi_head_attention_1/Mul/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€ї÷
$multi_head_attention_1/einsum/EinsumEinsum"multi_head_attention_1/key/add:z:0multi_head_attention_1/Mul:z:0*
N*
T0*1
_output_shapes
:€€€€€€€€€її*
equationaecd,abcd->acbeЬ
&multi_head_attention_1/softmax/SoftmaxSoftmax-multi_head_attention_1/einsum/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€її°
'multi_head_attention_1/dropout/IdentityIdentity0multi_head_attention_1/softmax/Softmax:softmax:0*
T0*1
_output_shapes
:€€€€€€€€€їїл
&multi_head_attention_1/einsum_1/EinsumEinsum0multi_head_attention_1/dropout/Identity:output:0$multi_head_attention_1/value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationacbe,aecd->abcd÷
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_1_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0Ы
5multi_head_attention_1/attention_output/einsum/EinsumEinsum/multi_head_attention_1/einsum_1/Einsum:output:0Lmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€ї*
equationabcd,cde->abeЇ
:multi_head_attention_1/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_1_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0п
+multi_head_attention_1/attention_output/addAddV2>multi_head_attention_1/attention_output/einsum/Einsum:output:0Bmulti_head_attention_1/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€їѓ
tf.__operators__.add_2/AddV2AddV2/multi_head_attention_1/attention_output/add:z:0 tf.__operators__.add_1/AddV2:z:0*
T0*,
_output_shapes
:€€€€€€€€€їi
conv1d_2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€Ѓ
conv1d_2/Conv1D/ExpandDims
ExpandDims tf.__operators__.add_2/AddV2:z:0'conv1d_2/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ї§
+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0b
 conv1d_2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ї
conv1d_2/Conv1D/ExpandDims_1
ExpandDims3conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@…
conv1d_2/Conv1DConv2D#conv1d_2/Conv1D/ExpandDims:output:0%conv1d_2/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ї@*
paddingVALID*
strides
У
conv1d_2/Conv1D/SqueezeSqueezeconv1d_2/Conv1D:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї@*
squeeze_dims

э€€€€€€€€Д
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Э
conv1d_2/BiasAddBiasAdd conv1d_2/Conv1D/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€ї@g
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї@r
dropout_1/IdentityIdentityconv1d_2/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€ї@i
conv1d_3/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€©
conv1d_3/Conv1D/ExpandDims
ExpandDimsdropout_1/Identity:output:0'conv1d_3/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ї@§
+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0b
 conv1d_3/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ї
conv1d_3/Conv1D/ExpandDims_1
ExpandDims3conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_3/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@…
conv1d_3/Conv1DConv2D#conv1d_3/Conv1D/ExpandDims:output:0%conv1d_3/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ї*
paddingVALID*
strides
У
conv1d_3/Conv1D/SqueezeSqueezeconv1d_3/Conv1D:output:0*
T0*,
_output_shapes
:€€€€€€€€€ї*
squeeze_dims

э€€€€€€€€Д
conv1d_3/BiasAdd/ReadVariableOpReadVariableOp(conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Э
conv1d_3/BiasAddBiasAdd conv1d_3/Conv1D/Squeeze:output:0'conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€їg
conv1d_3/ReluReluconv1d_3/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€їЫ
tf.__operators__.add_3/AddV2AddV2conv1d_3/Relu:activations:0 tf.__operators__.add_2/AddV2:z:0*
T0*,
_output_shapes
:€€€€€€€€€їl
*global_max_pooling1d/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :©
global_max_pooling1d/MaxMax tf.__operators__.add_3/AddV2:z:03global_max_pooling1d/Max/reduction_indices:output:0*
T0*(
_output_shapes
:€€€€€€€€€їt
dropout_2/IdentityIdentity!global_max_pooling1d/Max:output:0*
T0*(
_output_shapes
:€€€€€€€€€їБ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	ї@*
dtype0К
dense/MatMulMatMuldropout_2/Identity:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0И
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Д
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0Л
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@В
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0О
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Д
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Н
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€В
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€f
dense_2/SoftmaxSoftmaxdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€h
IdentityIdentitydense_2/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€≈
NoOpNoOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_2/BiasAdd/ReadVariableOp,^conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_3/BiasAdd/ReadVariableOp,^conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp;^multi_head_attention_1/attention_output/add/ReadVariableOpE^multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_1/key/add/ReadVariableOp8^multi_head_attention_1/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_1/query/add/ReadVariableOp:^multi_head_attention_1/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_1/value/add/ReadVariableOp:^multi_head_attention_1/value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:€€€€€€€€€ї: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/Conv1D/ExpandDims_1/ReadVariableOp)conv1d/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp2Z
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_2/BiasAdd/ReadVariableOpconv1d_2/BiasAdd/ReadVariableOp2Z
+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_3/BiasAdd/ReadVariableOpconv1d_3/BiasAdd/ReadVariableOp2Z
+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2t
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2И
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Z
+multi_head_attention/key/add/ReadVariableOp+multi_head_attention/key/add/ReadVariableOp2n
5multi_head_attention/key/einsum/Einsum/ReadVariableOp5multi_head_attention/key/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/query/add/ReadVariableOp-multi_head_attention/query/add/ReadVariableOp2r
7multi_head_attention/query/einsum/Einsum/ReadVariableOp7multi_head_attention/query/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/value/add/ReadVariableOp-multi_head_attention/value/add/ReadVariableOp2r
7multi_head_attention/value/einsum/Einsum/ReadVariableOp7multi_head_attention/value/einsum/Einsum/ReadVariableOp2x
:multi_head_attention_1/attention_output/add/ReadVariableOp:multi_head_attention_1/attention_output/add/ReadVariableOp2М
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_1/key/add/ReadVariableOp-multi_head_attention_1/key/add/ReadVariableOp2r
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOp7multi_head_attention_1/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_1/query/add/ReadVariableOp/multi_head_attention_1/query/add/ReadVariableOp2v
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_1/value/add/ReadVariableOp/multi_head_attention_1/value/add/ReadVariableOp2v
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€ї
 
_user_specified_nameinputs
ЊI
Ш
A__inference_model_layer_call_and_return_conditional_losses_189166
input_11
multi_head_attention_189088:-
multi_head_attention_189090:1
multi_head_attention_189092:-
multi_head_attention_189094:1
multi_head_attention_189096:-
multi_head_attention_189098:1
multi_head_attention_189100:)
multi_head_attention_189102:#
conv1d_189106:@
conv1d_189108:@%
conv1d_1_189112:@
conv1d_1_189114:3
multi_head_attention_1_189118:/
multi_head_attention_1_189120:3
multi_head_attention_1_189122:/
multi_head_attention_1_189124:3
multi_head_attention_1_189126:/
multi_head_attention_1_189128:3
multi_head_attention_1_189130:+
multi_head_attention_1_189132:%
conv1d_2_189136:@
conv1d_2_189138:@%
conv1d_3_189142:@
conv1d_3_189144:
dense_189150:	ї@
dense_189152:@ 
dense_1_189155:@@
dense_1_189157:@ 
dense_2_189160:@
dense_2_189162:
identityИҐconv1d/StatefulPartitionedCallҐ conv1d_1/StatefulPartitionedCallҐ conv1d_2/StatefulPartitionedCallҐ conv1d_3/StatefulPartitionedCallҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐdense_2/StatefulPartitionedCallҐ,multi_head_attention/StatefulPartitionedCallҐ.multi_head_attention_1/StatefulPartitionedCallн
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCallinput_1input_1multi_head_attention_189088multi_head_attention_189090multi_head_attention_189092multi_head_attention_189094multi_head_attention_189096multi_head_attention_189098multi_head_attention_189100multi_head_attention_189102*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ї**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_188196Ъ
tf.__operators__.add/AddV2AddV25multi_head_attention/StatefulPartitionedCall:output:0input_1*
T0*,
_output_shapes
:€€€€€€€€€їИ
conv1d/StatefulPartitionedCallStatefulPartitionedCalltf.__operators__.add/AddV2:z:0conv1d_189106conv1d_189108*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ї@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_188231я
dropout/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ї@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_188242Т
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0conv1d_1_189112conv1d_1_189114*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ї*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_188260І
tf.__operators__.add_1/AddV2AddV2)conv1d_1/StatefulPartitionedCall:output:0tf.__operators__.add/AddV2:z:0*
T0*,
_output_shapes
:€€€€€€€€€ї≥
.multi_head_attention_1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_1/AddV2:z:0 tf.__operators__.add_1/AddV2:z:0multi_head_attention_1_189118multi_head_attention_1_189120multi_head_attention_1_189122multi_head_attention_1_189124multi_head_attention_1_189126multi_head_attention_1_189128multi_head_attention_1_189130multi_head_attention_1_189132*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ї**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_188302Ј
tf.__operators__.add_2/AddV2AddV27multi_head_attention_1/StatefulPartitionedCall:output:0 tf.__operators__.add_1/AddV2:z:0*
T0*,
_output_shapes
:€€€€€€€€€їТ
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_2/AddV2:z:0conv1d_2_189136conv1d_2_189138*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ї@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_188337е
dropout_1/PartitionedCallPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ї@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_188348Ф
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0conv1d_3_189142conv1d_3_189144*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ї*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv1d_3_layer_call_and_return_conditional_losses_188366©
tf.__operators__.add_3/AddV2AddV2)conv1d_3/StatefulPartitionedCall:output:0 tf.__operators__.add_2/AddV2:z:0*
T0*,
_output_shapes
:€€€€€€€€€їо
$global_max_pooling1d/PartitionedCallPartitionedCall tf.__operators__.add_3/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ї* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_188151е
dropout_2/PartitionedCallPartitionedCall-global_max_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ї* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_188379Г
dense/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_189150dense_189152*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_188392П
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_189155dense_1_189157*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_188409С
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_189160dense_2_189162*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_188426w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Ф
NoOpNoOp^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCall/^multi_head_attention_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:€€€€€€€€€ї: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2\
,multi_head_attention/StatefulPartitionedCall,multi_head_attention/StatefulPartitionedCall2`
.multi_head_attention_1/StatefulPartitionedCall.multi_head_attention_1/StatefulPartitionedCall:U Q
,
_output_shapes
:€€€€€€€€€ї
!
_user_specified_name	input_1
√
Х
(__inference_dense_2_layer_call_fn_190252

inputs
unknown:@
	unknown_0:
identityИҐStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_188426o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
С2
щ
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_189890	
query	
valueA
+query_einsum_einsum_readvariableop_resource:3
!query_add_readvariableop_resource:?
)key_einsum_einsum_readvariableop_resource:1
key_add_readvariableop_resource:A
+value_einsum_einsum_readvariableop_resource:3
!value_add_readvariableop_resource:L
6attention_output_einsum_einsum_readvariableop_resource::
,attention_output_add_readvariableop_resource:
identityИҐ#attention_output/add/ReadVariableOpҐ-attention_output/einsum/Einsum/ReadVariableOpҐkey/add/ReadVariableOpҐ key/einsum/Einsum/ReadVariableOpҐquery/add/ReadVariableOpҐ"query/einsum/Einsum/ReadVariableOpҐvalue/add/ReadVariableOpҐ"value/einsum/Einsum/ReadVariableOpТ
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0±
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:*
dtype0Н
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€їО
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0≠
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:*
dtype0З
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€їТ
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0±
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:*
dtype0Н
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€їJ
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *уµ>d
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€їС
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:€€€€€€€€€її*
equationaecd,abcd->acben
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€їїZ
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?С
dropout/dropout/MulMulsoftmax/Softmax:softmax:0dropout/dropout/Const:output:0*
T0*1
_output_shapes
:€€€€€€€€€її^
dropout/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
:Ј
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*1
_output_shapes
:€€€€€€€€€її*
dtype0*
seed2€€€€c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=»
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:€€€€€€€€€їїЙ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:€€€€€€€€€їїЛ
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:€€€€€€€€€її¶
einsum_1/EinsumEinsumdropout/dropout/Mul_1:z:0value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ї*
equationacbe,aecd->abcd®
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0÷
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€ї*
equationabcd,cde->abeМ
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0™
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€їl
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€їЎ
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:€€€€€€€€€ї:€€€€€€€€€ї: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:€€€€€€€€€ї

_user_specified_namequery:SO
,
_output_shapes
:€€€€€€€€€ї

_user_specified_namevalue
¬
Ф
&__inference_dense_layer_call_fn_190212

inputs
unknown:	ї@
	unknown_0:@
identityИҐStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_188392o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ї: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€ї
 
_user_specified_nameinputs
№
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_190191

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:€€€€€€€€€ї\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:€€€€€€€€€ї"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€ї:P L
(
_output_shapes
:€€€€€€€€€ї
 
_user_specified_nameinputs"џL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ѓ
serving_defaultЫ
@
input_15
serving_default_input_1:0€€€€€€€€€ї;
dense_20
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:їЙ
њ
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer-10
layer_with_weights-5
layer-11
layer-12
layer-13
layer-14
layer_with_weights-6
layer-15
layer_with_weights-7
layer-16
layer_with_weights-8
layer-17
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
О
_query_dense

_key_dense
_value_dense
_softmax
 _dropout_layer
!_output_dense
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses"
_tf_keras_layer
(
(	keras_api"
_tf_keras_layer
ї

)kernel
*bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses"
_tf_keras_layer
Љ
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5_random_generator
6__call__
*7&call_and_return_all_conditional_losses"
_tf_keras_layer
ї

8kernel
9bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses"
_tf_keras_layer
(
@	keras_api"
_tf_keras_layer
О
A_query_dense
B
_key_dense
C_value_dense
D_softmax
E_dropout_layer
F_output_dense
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses"
_tf_keras_layer
(
M	keras_api"
_tf_keras_layer
ї

Nkernel
Obias
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses"
_tf_keras_layer
Љ
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z_random_generator
[__call__
*\&call_and_return_all_conditional_losses"
_tf_keras_layer
ї

]kernel
^bias
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses"
_tf_keras_layer
(
e	keras_api"
_tf_keras_layer
•
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses"
_tf_keras_layer
Љ
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p_random_generator
q__call__
*r&call_and_return_all_conditional_losses"
_tf_keras_layer
ї

skernel
tbias
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses"
_tf_keras_layer
Њ

{kernel
|bias
}	variables
~trainable_variables
regularization_losses
А	keras_api
Б__call__
+В&call_and_return_all_conditional_losses"
_tf_keras_layer
√
Гkernel
	Дbias
Е	variables
Жtrainable_variables
Зregularization_losses
И	keras_api
Й__call__
+К&call_and_return_all_conditional_losses"
_tf_keras_layer
‘
	Лiter
Мbeta_1
Нbeta_2

Оdecay
Пlearning_rate)mИ*mЙ8mК9mЛNmМOmН]mО^mПsmРtmС{mТ|mУ	ГmФ	ДmХ	РmЦ	СmЧ	ТmШ	УmЩ	ФmЪ	ХmЫ	ЦmЬ	ЧmЭ	ШmЮ	ЩmЯ	Ъm†	Ыm°	ЬmҐ	Эm£	Юm§	Яm•)v¶*vІ8v®9v©Nv™OvЂ]vђ^v≠svЃtvѓ{v∞|v±	Гv≤	Дv≥	Рvі	Сvµ	Тvґ	УvЈ	ФvЄ	Хvє	ЦvЇ	Чvї	ШvЉ	Щvљ	ЪvЊ	Ыvњ	Ьvј	ЭvЅ	Юv¬	Яv√"
	optimizer
Ш
Р0
С1
Т2
У3
Ф4
Х5
Ц6
Ч7
)8
*9
810
911
Ш12
Щ13
Ъ14
Ы15
Ь16
Э17
Ю18
Я19
N20
O21
]22
^23
s24
t25
{26
|27
Г28
Д29"
trackable_list_wrapper
Ш
Р0
С1
Т2
У3
Ф4
Х5
Ц6
Ч7
)8
*9
810
911
Ш12
Щ13
Ъ14
Ы15
Ь16
Э17
Ю18
Я19
N20
O21
]22
^23
s24
t25
{26
|27
Г28
Д29"
trackable_list_wrapper
 "
trackable_list_wrapper
ѕ
†non_trainable_variables
°layers
Ґmetrics
 £layer_regularization_losses
§layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ж2г
&__inference_model_layer_call_fn_188496
&__inference_model_layer_call_fn_189318
&__inference_model_layer_call_fn_189383
&__inference_model_layer_call_fn_189085ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
“2ѕ
A__inference_model_layer_call_and_return_conditional_losses_189525
A__inference_model_layer_call_and_return_conditional_losses_189702
A__inference_model_layer_call_and_return_conditional_losses_189166
A__inference_model_layer_call_and_return_conditional_losses_189247ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ћB…
!__inference__wrapped_model_188141input_1"Ш
С≤Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
-
•serving_default"
signature_map
ц
¶partial_output_shape
Іfull_output_shape
Рkernel
	Сbias
®	variables
©trainable_variables
™regularization_losses
Ђ	keras_api
ђ__call__
+≠&call_and_return_all_conditional_losses"
_tf_keras_layer
ц
Ѓpartial_output_shape
ѓfull_output_shape
Тkernel
	Уbias
∞	variables
±trainable_variables
≤regularization_losses
≥	keras_api
і__call__
+µ&call_and_return_all_conditional_losses"
_tf_keras_layer
ц
ґpartial_output_shape
Јfull_output_shape
Фkernel
	Хbias
Є	variables
єtrainable_variables
Їregularization_losses
ї	keras_api
Љ__call__
+љ&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
Њ	variables
њtrainable_variables
јregularization_losses
Ѕ	keras_api
¬__call__
+√&call_and_return_all_conditional_losses"
_tf_keras_layer
√
ƒ	variables
≈trainable_variables
∆regularization_losses
«	keras_api
»_random_generator
…__call__
+ &call_and_return_all_conditional_losses"
_tf_keras_layer
ц
Ћpartial_output_shape
ћfull_output_shape
Цkernel
	Чbias
Ќ	variables
ќtrainable_variables
ѕregularization_losses
–	keras_api
—__call__
+“&call_and_return_all_conditional_losses"
_tf_keras_layer
`
Р0
С1
Т2
У3
Ф4
Х5
Ц6
Ч7"
trackable_list_wrapper
`
Р0
С1
Т2
У3
Ф4
Х5
Ц6
Ч7"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
”non_trainable_variables
‘layers
’metrics
 ÷layer_regularization_losses
„layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
р2н
5__inference_multi_head_attention_layer_call_fn_189791
5__inference_multi_head_attention_layer_call_fn_189813ь
у≤п
FullArgSpece
args]ЪZ
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaultsЪ

 

 
p 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
¶2£
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_189848
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_189890ь
у≤п
FullArgSpece
args]ЪZ
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaultsЪ

 

 
p 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
"
_generic_user_object
#:!@2conv1d/kernel
:@2conv1d/bias
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Ўnon_trainable_variables
ўlayers
Џmetrics
 џlayer_regularization_losses
№layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
—2ќ
'__inference_conv1d_layer_call_fn_189899Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
м2й
B__inference_conv1d_layer_call_and_return_conditional_losses_189915Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Ёnon_trainable_variables
ёlayers
яmetrics
 аlayer_regularization_losses
бlayer_metrics
1	variables
2trainable_variables
3regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
О2Л
(__inference_dropout_layer_call_fn_189920
(__inference_dropout_layer_call_fn_189925і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ƒ2Ѕ
C__inference_dropout_layer_call_and_return_conditional_losses_189930
C__inference_dropout_layer_call_and_return_conditional_losses_189942і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
%:#@2conv1d_1/kernel
:2conv1d_1/bias
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
вnon_trainable_variables
гlayers
дmetrics
 еlayer_regularization_losses
жlayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
”2–
)__inference_conv1d_1_layer_call_fn_189951Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
о2л
D__inference_conv1d_1_layer_call_and_return_conditional_losses_189967Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
"
_generic_user_object
ц
зpartial_output_shape
иfull_output_shape
Шkernel
	Щbias
й	variables
кtrainable_variables
лregularization_losses
м	keras_api
н__call__
+о&call_and_return_all_conditional_losses"
_tf_keras_layer
ц
пpartial_output_shape
рfull_output_shape
Ъkernel
	Ыbias
с	variables
тtrainable_variables
уregularization_losses
ф	keras_api
х__call__
+ц&call_and_return_all_conditional_losses"
_tf_keras_layer
ц
чpartial_output_shape
шfull_output_shape
Ьkernel
	Эbias
щ	variables
ъtrainable_variables
ыregularization_losses
ь	keras_api
э__call__
+ю&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
€	variables
Аtrainable_variables
Бregularization_losses
В	keras_api
Г__call__
+Д&call_and_return_all_conditional_losses"
_tf_keras_layer
√
Е	variables
Жtrainable_variables
Зregularization_losses
И	keras_api
Й_random_generator
К__call__
+Л&call_and_return_all_conditional_losses"
_tf_keras_layer
ц
Мpartial_output_shape
Нfull_output_shape
Юkernel
	Яbias
О	variables
Пtrainable_variables
Рregularization_losses
С	keras_api
Т__call__
+У&call_and_return_all_conditional_losses"
_tf_keras_layer
`
Ш0
Щ1
Ъ2
Ы3
Ь4
Э5
Ю6
Я7"
trackable_list_wrapper
`
Ш0
Щ1
Ъ2
Ы3
Ь4
Э5
Ю6
Я7"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Фnon_trainable_variables
Хlayers
Цmetrics
 Чlayer_regularization_losses
Шlayer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
ф2с
7__inference_multi_head_attention_1_layer_call_fn_189989
7__inference_multi_head_attention_1_layer_call_fn_190011ь
у≤п
FullArgSpece
args]ЪZ
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaultsЪ

 

 
p 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
™2І
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_190046
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_190088ь
у≤п
FullArgSpece
args]ЪZ
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaultsЪ

 

 
p 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
"
_generic_user_object
%:#@2conv1d_2/kernel
:@2conv1d_2/bias
.
N0
O1"
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Щnon_trainable_variables
Ъlayers
Ыmetrics
 Ьlayer_regularization_losses
Эlayer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
”2–
)__inference_conv1d_2_layer_call_fn_190097Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
о2л
D__inference_conv1d_2_layer_call_and_return_conditional_losses_190113Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Юnon_trainable_variables
Яlayers
†metrics
 °layer_regularization_losses
Ґlayer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
Т2П
*__inference_dropout_1_layer_call_fn_190118
*__inference_dropout_1_layer_call_fn_190123і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
»2≈
E__inference_dropout_1_layer_call_and_return_conditional_losses_190128
E__inference_dropout_1_layer_call_and_return_conditional_losses_190140і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
%:#@2conv1d_3/kernel
:2conv1d_3/bias
.
]0
^1"
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
£non_trainable_variables
§layers
•metrics
 ¶layer_regularization_losses
Іlayer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
”2–
)__inference_conv1d_3_layer_call_fn_190149Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
о2л
D__inference_conv1d_3_layer_call_and_return_conditional_losses_190165Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
®non_trainable_variables
©layers
™metrics
 Ђlayer_regularization_losses
ђlayer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
я2№
5__inference_global_max_pooling1d_layer_call_fn_190170Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ъ2ч
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_190176Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
≠non_trainable_variables
Ѓlayers
ѓmetrics
 ∞layer_regularization_losses
±layer_metrics
l	variables
mtrainable_variables
nregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
Т2П
*__inference_dropout_2_layer_call_fn_190181
*__inference_dropout_2_layer_call_fn_190186і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
»2≈
E__inference_dropout_2_layer_call_and_return_conditional_losses_190191
E__inference_dropout_2_layer_call_and_return_conditional_losses_190203і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
:	ї@2dense/kernel
:@2
dense/bias
.
s0
t1"
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
≤non_trainable_variables
≥layers
іmetrics
 µlayer_regularization_losses
ґlayer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
–2Ќ
&__inference_dense_layer_call_fn_190212Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
л2и
A__inference_dense_layer_call_and_return_conditional_losses_190223Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 :@@2dense_1/kernel
:@2dense_1/bias
.
{0
|1"
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Јnon_trainable_variables
Єlayers
єmetrics
 Їlayer_regularization_losses
їlayer_metrics
}	variables
~trainable_variables
regularization_losses
Б__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses"
_generic_user_object
“2ѕ
(__inference_dense_1_layer_call_fn_190232Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
н2к
C__inference_dense_1_layer_call_and_return_conditional_losses_190243Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 :@2dense_2/kernel
:2dense_2/bias
0
Г0
Д1"
trackable_list_wrapper
0
Г0
Д1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Љnon_trainable_variables
љlayers
Њmetrics
 њlayer_regularization_losses
јlayer_metrics
Е	variables
Жtrainable_variables
Зregularization_losses
Й__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses"
_generic_user_object
“2ѕ
(__inference_dense_2_layer_call_fn_190252Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
н2к
C__inference_dense_2_layer_call_and_return_conditional_losses_190263Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
7:52!multi_head_attention/query/kernel
1:/2multi_head_attention/query/bias
5:32multi_head_attention/key/kernel
/:-2multi_head_attention/key/bias
7:52!multi_head_attention/value/kernel
1:/2multi_head_attention/value/bias
B:@2,multi_head_attention/attention_output/kernel
8:62*multi_head_attention/attention_output/bias
9:72#multi_head_attention_1/query/kernel
3:12!multi_head_attention_1/query/bias
7:52!multi_head_attention_1/key/kernel
1:/2multi_head_attention_1/key/bias
9:72#multi_head_attention_1/value/kernel
3:12!multi_head_attention_1/value/bias
D:B2.multi_head_attention_1/attention_output/kernel
::82,multi_head_attention_1/attention_output/bias
 "
trackable_list_wrapper
¶
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17"
trackable_list_wrapper
0
Ѕ0
¬1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЋB»
$__inference_signature_wrapper_189769input_1"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
Р0
С1"
trackable_list_wrapper
0
Р0
С1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
√non_trainable_variables
ƒlayers
≈metrics
 ∆layer_regularization_losses
«layer_metrics
®	variables
©trainable_variables
™regularization_losses
ђ__call__
+≠&call_and_return_all_conditional_losses
'≠"call_and_return_conditional_losses"
_generic_user_object
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
Т0
У1"
trackable_list_wrapper
0
Т0
У1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
»non_trainable_variables
…layers
 metrics
 Ћlayer_regularization_losses
ћlayer_metrics
∞	variables
±trainable_variables
≤regularization_losses
і__call__
+µ&call_and_return_all_conditional_losses
'µ"call_and_return_conditional_losses"
_generic_user_object
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ф0
Х1"
trackable_list_wrapper
0
Ф0
Х1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Ќnon_trainable_variables
ќlayers
ѕmetrics
 –layer_regularization_losses
—layer_metrics
Є	variables
єtrainable_variables
Їregularization_losses
Љ__call__
+љ&call_and_return_all_conditional_losses
'љ"call_and_return_conditional_losses"
_generic_user_object
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
“non_trainable_variables
”layers
‘metrics
 ’layer_regularization_losses
÷layer_metrics
Њ	variables
њtrainable_variables
јregularization_losses
¬__call__
+√&call_and_return_all_conditional_losses
'√"call_and_return_conditional_losses"
_generic_user_object
µ2≤ѓ
¶≤Ґ
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsҐ

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
µ2≤ѓ
¶≤Ґ
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsҐ

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
„non_trainable_variables
Ўlayers
ўmetrics
 Џlayer_regularization_losses
џlayer_metrics
ƒ	variables
≈trainable_variables
∆regularization_losses
…__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ц0
Ч1"
trackable_list_wrapper
0
Ц0
Ч1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
№non_trainable_variables
Ёlayers
ёmetrics
 яlayer_regularization_losses
аlayer_metrics
Ќ	variables
ќtrainable_variables
ѕregularization_losses
—__call__
+“&call_and_return_all_conditional_losses
'“"call_and_return_conditional_losses"
_generic_user_object
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
J
0
1
2
3
 4
!5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ш0
Щ1"
trackable_list_wrapper
0
Ш0
Щ1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
бnon_trainable_variables
вlayers
гmetrics
 дlayer_regularization_losses
еlayer_metrics
й	variables
кtrainable_variables
лregularization_losses
н__call__
+о&call_and_return_all_conditional_losses
'о"call_and_return_conditional_losses"
_generic_user_object
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ъ0
Ы1"
trackable_list_wrapper
0
Ъ0
Ы1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
жnon_trainable_variables
зlayers
иmetrics
 йlayer_regularization_losses
кlayer_metrics
с	variables
тtrainable_variables
уregularization_losses
х__call__
+ц&call_and_return_all_conditional_losses
'ц"call_and_return_conditional_losses"
_generic_user_object
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ь0
Э1"
trackable_list_wrapper
0
Ь0
Э1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
лnon_trainable_variables
мlayers
нmetrics
 оlayer_regularization_losses
пlayer_metrics
щ	variables
ъtrainable_variables
ыregularization_losses
э__call__
+ю&call_and_return_all_conditional_losses
'ю"call_and_return_conditional_losses"
_generic_user_object
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
рnon_trainable_variables
сlayers
тmetrics
 уlayer_regularization_losses
фlayer_metrics
€	variables
Аtrainable_variables
Бregularization_losses
Г__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
_generic_user_object
µ2≤ѓ
¶≤Ґ
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsҐ

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
µ2≤ѓ
¶≤Ґ
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsҐ

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
хnon_trainable_variables
цlayers
чmetrics
 шlayer_regularization_losses
щlayer_metrics
Е	variables
Жtrainable_variables
Зregularization_losses
К__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ю0
Я1"
trackable_list_wrapper
0
Ю0
Я1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
ъnon_trainable_variables
ыlayers
ьmetrics
 эlayer_regularization_losses
юlayer_metrics
О	variables
Пtrainable_variables
Рregularization_losses
Т__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses"
_generic_user_object
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
J
A0
B1
C2
D3
E4
F5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

€total

Аcount
Б	variables
В	keras_api"
_tf_keras_metric
c

Гtotal

Дcount
Е
_fn_kwargs
Ж	variables
З	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
€0
А1"
trackable_list_wrapper
.
Б	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
Г0
Д1"
trackable_list_wrapper
.
Ж	variables"
_generic_user_object
(:&@2Adam/conv1d/kernel/m
:@2Adam/conv1d/bias/m
*:(@2Adam/conv1d_1/kernel/m
 :2Adam/conv1d_1/bias/m
*:(@2Adam/conv1d_2/kernel/m
 :@2Adam/conv1d_2/bias/m
*:(@2Adam/conv1d_3/kernel/m
 :2Adam/conv1d_3/bias/m
$:"	ї@2Adam/dense/kernel/m
:@2Adam/dense/bias/m
%:#@@2Adam/dense_1/kernel/m
:@2Adam/dense_1/bias/m
%:#@2Adam/dense_2/kernel/m
:2Adam/dense_2/bias/m
<::2(Adam/multi_head_attention/query/kernel/m
6:42&Adam/multi_head_attention/query/bias/m
::82&Adam/multi_head_attention/key/kernel/m
4:22$Adam/multi_head_attention/key/bias/m
<::2(Adam/multi_head_attention/value/kernel/m
6:42&Adam/multi_head_attention/value/bias/m
G:E23Adam/multi_head_attention/attention_output/kernel/m
=:;21Adam/multi_head_attention/attention_output/bias/m
>:<2*Adam/multi_head_attention_1/query/kernel/m
8:62(Adam/multi_head_attention_1/query/bias/m
<::2(Adam/multi_head_attention_1/key/kernel/m
6:42&Adam/multi_head_attention_1/key/bias/m
>:<2*Adam/multi_head_attention_1/value/kernel/m
8:62(Adam/multi_head_attention_1/value/bias/m
I:G25Adam/multi_head_attention_1/attention_output/kernel/m
?:=23Adam/multi_head_attention_1/attention_output/bias/m
(:&@2Adam/conv1d/kernel/v
:@2Adam/conv1d/bias/v
*:(@2Adam/conv1d_1/kernel/v
 :2Adam/conv1d_1/bias/v
*:(@2Adam/conv1d_2/kernel/v
 :@2Adam/conv1d_2/bias/v
*:(@2Adam/conv1d_3/kernel/v
 :2Adam/conv1d_3/bias/v
$:"	ї@2Adam/dense/kernel/v
:@2Adam/dense/bias/v
%:#@@2Adam/dense_1/kernel/v
:@2Adam/dense_1/bias/v
%:#@2Adam/dense_2/kernel/v
:2Adam/dense_2/bias/v
<::2(Adam/multi_head_attention/query/kernel/v
6:42&Adam/multi_head_attention/query/bias/v
::82&Adam/multi_head_attention/key/kernel/v
4:22$Adam/multi_head_attention/key/bias/v
<::2(Adam/multi_head_attention/value/kernel/v
6:42&Adam/multi_head_attention/value/bias/v
G:E23Adam/multi_head_attention/attention_output/kernel/v
=:;21Adam/multi_head_attention/attention_output/bias/v
>:<2*Adam/multi_head_attention_1/query/kernel/v
8:62(Adam/multi_head_attention_1/query/bias/v
<::2(Adam/multi_head_attention_1/key/kernel/v
6:42&Adam/multi_head_attention_1/key/bias/v
>:<2*Adam/multi_head_attention_1/value/kernel/v
8:62(Adam/multi_head_attention_1/value/bias/v
I:G25Adam/multi_head_attention_1/attention_output/kernel/v
?:=23Adam/multi_head_attention_1/attention_output/bias/v¬
!__inference__wrapped_model_188141Ь0РСТУФХЦЧ)*89ШЩЪЫЬЭЮЯNO]^st{|ГД5Ґ2
+Ґ(
&К#
input_1€€€€€€€€€ї
™ "1™.
,
dense_2!К
dense_2€€€€€€€€€Ѓ
D__inference_conv1d_1_layer_call_and_return_conditional_losses_189967f894Ґ1
*Ґ'
%К"
inputs€€€€€€€€€ї@
™ "*Ґ'
 К
0€€€€€€€€€ї
Ъ Ж
)__inference_conv1d_1_layer_call_fn_189951Y894Ґ1
*Ґ'
%К"
inputs€€€€€€€€€ї@
™ "К€€€€€€€€€їЃ
D__inference_conv1d_2_layer_call_and_return_conditional_losses_190113fNO4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€ї
™ "*Ґ'
 К
0€€€€€€€€€ї@
Ъ Ж
)__inference_conv1d_2_layer_call_fn_190097YNO4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€ї
™ "К€€€€€€€€€ї@Ѓ
D__inference_conv1d_3_layer_call_and_return_conditional_losses_190165f]^4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€ї@
™ "*Ґ'
 К
0€€€€€€€€€ї
Ъ Ж
)__inference_conv1d_3_layer_call_fn_190149Y]^4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€ї@
™ "К€€€€€€€€€їђ
B__inference_conv1d_layer_call_and_return_conditional_losses_189915f)*4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€ї
™ "*Ґ'
 К
0€€€€€€€€€ї@
Ъ Д
'__inference_conv1d_layer_call_fn_189899Y)*4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€ї
™ "К€€€€€€€€€ї@£
C__inference_dense_1_layer_call_and_return_conditional_losses_190243\{|/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "%Ґ"
К
0€€€€€€€€€@
Ъ {
(__inference_dense_1_layer_call_fn_190232O{|/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "К€€€€€€€€€@•
C__inference_dense_2_layer_call_and_return_conditional_losses_190263^ГД/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "%Ґ"
К
0€€€€€€€€€
Ъ }
(__inference_dense_2_layer_call_fn_190252QГД/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "К€€€€€€€€€Ґ
A__inference_dense_layer_call_and_return_conditional_losses_190223]st0Ґ-
&Ґ#
!К
inputs€€€€€€€€€ї
™ "%Ґ"
К
0€€€€€€€€€@
Ъ z
&__inference_dense_layer_call_fn_190212Pst0Ґ-
&Ґ#
!К
inputs€€€€€€€€€ї
™ "К€€€€€€€€€@ѓ
E__inference_dropout_1_layer_call_and_return_conditional_losses_190128f8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ї@
p 
™ "*Ґ'
 К
0€€€€€€€€€ї@
Ъ ѓ
E__inference_dropout_1_layer_call_and_return_conditional_losses_190140f8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ї@
p
™ "*Ґ'
 К
0€€€€€€€€€ї@
Ъ З
*__inference_dropout_1_layer_call_fn_190118Y8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ї@
p 
™ "К€€€€€€€€€ї@З
*__inference_dropout_1_layer_call_fn_190123Y8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ї@
p
™ "К€€€€€€€€€ї@І
E__inference_dropout_2_layer_call_and_return_conditional_losses_190191^4Ґ1
*Ґ'
!К
inputs€€€€€€€€€ї
p 
™ "&Ґ#
К
0€€€€€€€€€ї
Ъ І
E__inference_dropout_2_layer_call_and_return_conditional_losses_190203^4Ґ1
*Ґ'
!К
inputs€€€€€€€€€ї
p
™ "&Ґ#
К
0€€€€€€€€€ї
Ъ 
*__inference_dropout_2_layer_call_fn_190181Q4Ґ1
*Ґ'
!К
inputs€€€€€€€€€ї
p 
™ "К€€€€€€€€€ї
*__inference_dropout_2_layer_call_fn_190186Q4Ґ1
*Ґ'
!К
inputs€€€€€€€€€ї
p
™ "К€€€€€€€€€ї≠
C__inference_dropout_layer_call_and_return_conditional_losses_189930f8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ї@
p 
™ "*Ґ'
 К
0€€€€€€€€€ї@
Ъ ≠
C__inference_dropout_layer_call_and_return_conditional_losses_189942f8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ї@
p
™ "*Ґ'
 К
0€€€€€€€€€ї@
Ъ Е
(__inference_dropout_layer_call_fn_189920Y8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ї@
p 
™ "К€€€€€€€€€ї@Е
(__inference_dropout_layer_call_fn_189925Y8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ї@
p
™ "К€€€€€€€€€ї@Ћ
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_190176wEҐB
;Ґ8
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ".Ґ+
$К!
0€€€€€€€€€€€€€€€€€€
Ъ £
5__inference_global_max_pooling1d_layer_call_fn_190170jEҐB
;Ґ8
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "!К€€€€€€€€€€€€€€€€€€ё
A__inference_model_layer_call_and_return_conditional_losses_189166Ш0РСТУФХЦЧ)*89ШЩЪЫЬЭЮЯNO]^st{|ГД=Ґ:
3Ґ0
&К#
input_1€€€€€€€€€ї
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ё
A__inference_model_layer_call_and_return_conditional_losses_189247Ш0РСТУФХЦЧ)*89ШЩЪЫЬЭЮЯNO]^st{|ГД=Ґ:
3Ґ0
&К#
input_1€€€€€€€€€ї
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Ё
A__inference_model_layer_call_and_return_conditional_losses_189525Ч0РСТУФХЦЧ)*89ШЩЪЫЬЭЮЯNO]^st{|ГД<Ґ9
2Ґ/
%К"
inputs€€€€€€€€€ї
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Ё
A__inference_model_layer_call_and_return_conditional_losses_189702Ч0РСТУФХЦЧ)*89ШЩЪЫЬЭЮЯNO]^st{|ГД<Ґ9
2Ґ/
%К"
inputs€€€€€€€€€ї
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ґ
&__inference_model_layer_call_fn_188496Л0РСТУФХЦЧ)*89ШЩЪЫЬЭЮЯNO]^st{|ГД=Ґ:
3Ґ0
&К#
input_1€€€€€€€€€ї
p 

 
™ "К€€€€€€€€€ґ
&__inference_model_layer_call_fn_189085Л0РСТУФХЦЧ)*89ШЩЪЫЬЭЮЯNO]^st{|ГД=Ґ:
3Ґ0
&К#
input_1€€€€€€€€€ї
p

 
™ "К€€€€€€€€€µ
&__inference_model_layer_call_fn_189318К0РСТУФХЦЧ)*89ШЩЪЫЬЭЮЯNO]^st{|ГД<Ґ9
2Ґ/
%К"
inputs€€€€€€€€€ї
p 

 
™ "К€€€€€€€€€µ
&__inference_model_layer_call_fn_189383К0РСТУФХЦЧ)*89ШЩЪЫЬЭЮЯNO]^st{|ГД<Ґ9
2Ґ/
%К"
inputs€€€€€€€€€ї
p

 
™ "К€€€€€€€€€А
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_190046©ШЩЪЫЬЭЮЯiҐf
_Ґ\
$К!
query€€€€€€€€€ї
$К!
value€€€€€€€€€ї

 

 
p 
p 
™ "*Ґ'
 К
0€€€€€€€€€ї
Ъ А
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_190088©ШЩЪЫЬЭЮЯiҐf
_Ґ\
$К!
query€€€€€€€€€ї
$К!
value€€€€€€€€€ї

 

 
p 
p
™ "*Ґ'
 К
0€€€€€€€€€ї
Ъ Ў
7__inference_multi_head_attention_1_layer_call_fn_189989ЬШЩЪЫЬЭЮЯiҐf
_Ґ\
$К!
query€€€€€€€€€ї
$К!
value€€€€€€€€€ї

 

 
p 
p 
™ "К€€€€€€€€€їЎ
7__inference_multi_head_attention_1_layer_call_fn_190011ЬШЩЪЫЬЭЮЯiҐf
_Ґ\
$К!
query€€€€€€€€€ї
$К!
value€€€€€€€€€ї

 

 
p 
p
™ "К€€€€€€€€€їю
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_189848©РСТУФХЦЧiҐf
_Ґ\
$К!
query€€€€€€€€€ї
$К!
value€€€€€€€€€ї

 

 
p 
p 
™ "*Ґ'
 К
0€€€€€€€€€ї
Ъ ю
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_189890©РСТУФХЦЧiҐf
_Ґ\
$К!
query€€€€€€€€€ї
$К!
value€€€€€€€€€ї

 

 
p 
p
™ "*Ґ'
 К
0€€€€€€€€€ї
Ъ ÷
5__inference_multi_head_attention_layer_call_fn_189791ЬРСТУФХЦЧiҐf
_Ґ\
$К!
query€€€€€€€€€ї
$К!
value€€€€€€€€€ї

 

 
p 
p 
™ "К€€€€€€€€€ї÷
5__inference_multi_head_attention_layer_call_fn_189813ЬРСТУФХЦЧiҐf
_Ґ\
$К!
query€€€€€€€€€ї
$К!
value€€€€€€€€€ї

 

 
p 
p
™ "К€€€€€€€€€ї–
$__inference_signature_wrapper_189769І0РСТУФХЦЧ)*89ШЩЪЫЬЭЮЯNO]^st{|ГД@Ґ=
Ґ 
6™3
1
input_1&К#
input_1€€€€€€€€€ї"1™.
,
dense_2!К
dense_2€€€€€€€€€