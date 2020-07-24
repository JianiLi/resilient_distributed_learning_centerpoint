## Resilient distributed learning using centerpoint-based aggregation
```
Using centerpoint based aggregation for resilient distributed machine learning algorithms in Byzantine environment. 
```


### Examples
#### Linear regression
```
- static_target.py is a target pursuit example in mobile multi-robot networks, with static target.
- dynamic_target.py is a target pursuit example in mobile multi-robot networks, with dynamic target.
```
![](https://github.com/JianiLi/resilient_distributed_learning_centerpoint/blob/master/fig/simulation.gif)

It can be found only centepoint based rule converges to the target under attack, whereas all the other rules failed.

#### Logistic regression
```
- classfication.py.

In the case of 3 attackers in a complete network of 10 agents, using different aggregation rules, the decision boundary
is as follows (from left to right: average, coordinate-wise median, geometric median, centerpoint).
```
<img src="https://github.com/JianiLi/resilient_distributed_learning_centerpoint/blob/master/fig/clfresults_attacked3_randomoutlier_average.png" alt="drawing" width="200"/> <img src="https://github.com/JianiLi/resilient_distributed_learning_centerpoint/blob/master/fig/clfresults_attacked3_randomoutlier_CM.png" alt="drawing" width="200"/> <img src="https://github.com/JianiLi/resilient_distributed_learning_centerpoint/blob/master/fig/clfresults_attacked3_randomoutlier_GM.png" alt="drawing" width="200"/> <img src="https://github.com/JianiLi/resilient_distributed_learning_centerpoint/blob/master/fig/clfresults_attacked3_randomoutlier_centerpoint.png" alt="drawing" width="200"/>

It can be found centerpoint outperforms other rules.

### Cite the paper
```
The paper is under review by IEEE Transactions on Robotics (T-RO). 
The preliminary results can be found in the following conference paper:

@inproceedings{centerpoint_RSS,  
  title={Resilient Distributed Diffusion for Multi-Robot Systems Using Centerpoint},  
  author={J. Li and W. Abbas and M. Shabbir and X. Koutsoukos},  
  booktitle = {Robotics: Science and Systems},  
  year      = {2020}  
}
```
