## 2021.04.10

### 학습자료
[위키독스 : PyTorch로 시작하는 딥 러닝 입문](https://wikidocs.net/book/2788)


### 선형 회귀

-----------------------------

 + 1. 데이터에 대한 이해 
    > 훈련 데이터 셋을 구축 (x= (1,2,3) 으로 구성될때 y = (2,4,6) 형식)

 + 2. 가설 수립
    > y = Wx + b 라는 식을 세운다. (W : 가중치, b : 편향)

 + 3. 비용함수에 대한 이해 
    > 평균 제곱 오차(MSE)를 이용하여 비용함수를 최소가 되게 만드는 W와 b를 구하는 것이 목표이다

 + 4. Optimizer - Gradient Desent
    > 비용함수가 최소가 되는 부분은 접선의 기울기가 0인 지점이다.
    > 기울기가 음수일땐 W값이 증가, 양수일땐 W값이 감소 한다.
    

### Pytorch로 구현하기

--------------------

```buildoutcfg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 데이터
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

# 모델 초기화
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer 설정
optimizer = optim.SGD([W, b], lr=0.01)

nb_epochs = 1999 # 원하는만큼 경사 하강법을 반복
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    hypothesis = x_train * W + b

    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, W.item(), b.item(), cost.item()
        ))
```

<br>

### Logistic 회귀 

-------------------------------------------------

 + 둘중 하나를 결정하는 이진 분류(binary Classification)
    > 값이 1과 0으로 나누어 진다.
      예 : step function (x < 0 이면 y = 0 x > 0 이면 y = 1)
    
 + 시그모이드 함수 (sigmoid function)
    > 가장 많이 쓰이는 S자 형태의 그래프이다. x가 매우 작으면 0에 수렴하고 x가 매우크면 1에 수렴한다.
      x = 0일때에는 0.5이다 W에 따라 경사도가 바뀌고 b에 의해 그래프가 이동한다.
   
 + 비용함수
   > 시그모이드 함수를 사용하여 MSE를 사용하면 Local minima에 빠질 수도 있다.
     Momentum이나 Adagrad를 활용하여 비용함수를 구할 수 있다.
   > 

### Pytorch로 Logistic 회귀 구현하기

-------------------------

```buildoutcfg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# 데이터
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

# 모델 초기화
W = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # Cost 계산
    hypothesis = torch.sigmoid(x_train.matmul(W) + b)
    cost = -(y_train * torch.log(hypothesis) + (1 - y_train) * torch.log(1 - hypothesis)).mean()

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))
```