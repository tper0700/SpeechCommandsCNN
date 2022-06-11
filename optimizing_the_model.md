Before update to model

=> [0] finished...
model: 12.202 seconds
lossfn: 0.387 seconds
backward: 16.839 seconds
optimizer: 1.727 seconds
train time: 31.178 seconds
[[3.5467116386009185]]
[32.64496636390686]
[0.04861426624261699]
[535]

Remove the 1 dimension

=> [0] finished...
model: 11.104 seconds
lossfn: 0.431 seconds
backward: 14.224 seconds
optimizer: 2.077 seconds
train time: 27.872 seconds
[[3.5472650615292474]]
[29.603765726089478]
[0.04788732394366197]
[527]

JIT the function #1
=> [0] finished...
model: 11.113 seconds
lossfn: 0.398 seconds
backward: 16.819 seconds
optimizer: 2.544 seconds
train time: 30.904 seconds
[[3.5474420871092565]]
[32.50470519065857]
[0.0489777373920945]
[539]

JIT the whole thing
=> [0] finished...
model: 10.849 seconds
lossfn: 0.336 seconds
backward: 14.675 seconds
optimizer: 2.643 seconds
train time: 28.546 seconds
[[3.547612940319643]]
[30.448246717453003]
[0.05134029986369832]
[565]

using state.view instead of reshaping
=> [0] finished...
model: 10.871 seconds
lossfn: 0.478 seconds
backward: 14.451 seconds
optimizer: 2.693 seconds
train time: 28.535 seconds
[[3.5470106064196307]]
[30.460528135299683]
[0.05024988641526579]
[553]
