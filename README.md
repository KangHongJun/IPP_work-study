# IPP_work-study

## 1. [파이토치 튜툐리얼](https://github.com/KangHongJun/IPP_Xiilab/tree/main/pytorch)

### 진행과정
 [pytorch 튜툐리얼](https://tutorials.pytorch.kr/beginner/basics/quickstart_tutorial.html)을 Pycharm환경에서 conda 인터프리터로 설정하여 진행했다. 

2021년도에 한국IT비즈니스진흥협회에서 고급 딥러닝 과정을 들었기에 용어가 친숙했지만, 까먹은 상태였기 때문에 용어와 함수들의 의미에 대해 인지하는 과정이었다.



## 2. [yolov5를 이용한 재난 감지](https://github.com/KangHongJun/IPP_Xiilab/tree/main/collapse_data_train)

### 진행과정
[yolov5](https://github.com/ultralytics/yolov5)로 이미지/영상에서 사람&불&건물 붕괴를 인식할 수 있도록 해봤다.

 라이브러리는 사용해봤지만, 오픈소스를 clone하여 사용해보는것은 처음이었기 때문에 requirements.txt로 환경설정하는것을 처음 해봐서 신선한 경험이었고,<br>
pytorch 튜툐리얼을 진행하던 환경에 추가로 세팅하려니 라이브러리들이 서로 버전 충돌이 나서 갈아엎으면서 진행했다.<br>
 환경세팅이 끝난 후에는 yolov5의 공식문서에서 커스텀 데이터 훈련방법을 보면서 roboflow에서 fire 데이터를 다운받고 훈련시킨 후 이미지/영상을 넣어보니 불을 잘 감지했다.
그 다음에는 사람과 건물 붕괴 데이터도 넣어서 학습해봤지만, 컴퓨터 성능문제로 훈현을 조금 밖에 돌리지 못하여 성능이 좋지 않았다.




## 3. [Origin-NMS](https://github.com/KangHongJun/Origin-NMS)

### 진행과정

 먼저 오픈소스 [sahi](https://github.com/obss/sahi)를 detect model을 yolov5로 하여 세팅했다.
 [sahi라는 오픈소스의 목적과 문제점에 대해 파악하고 내장되어 있는 merge 알고리즘 NMS. NMM. GREEDYNMM을 하나씩 돌려보고 정리했고](https://github.com/KangHongJun/IPP_Xiilab/tree/main/sahi_improve/search) <br>
정리한 것을 바탕으로 피드백을 받았고, sahi의 문제점을 개선하기 위해 첫번째로 merge연산 진행시 IOU연산이 아닌 다른 연산을 하는 눈문에 대한 정보를 탐색해봤지만 공개한 것이 없어서 찾기 힘들었다. 그래서 먼저 겹침정도를 연산하는 IOU와 IOS에 대해 이해하고 NMS와IOS기반으로 개선해나가는것으로 방향을 잡았다.<br>
먼저 그냥 NMS를 진행하면 사라지는 box가 너무 많기 때문에 오리지널 box를 복구하는 방향으로 가자고 피드백을 받아서 진행하였고<br> 
결과는 box를 복구한 만큼 겹치는 부분이 많아졌기 때문에 그것을 해소할 방법이 필요했고, <br>
nms 진행시 최고score가 아닌 오리지널 box score기준으로 nms를 진행하였지만 사소한 변화만 있었다.
그리고 이번엔 슬라이싱 이미지의 box끼리 먼저 nms를 하고 그 다음에 오리지널 이미지의 box를 합쳐서 nms를 하는 방식으로 하자는 피드백을 받고 진행시켰더니
괜찮은 결과물을 얻어낼 수 있었다.

중간중간에 문법에 혼동이 있어서 코드를 잘못 작성한다거나, 결과 이미지 이름을 애매하게 작성하여 나중에 검토 할 때 어떤 코드에서 나온 결과물인지 모르는 상황이 나왔기 때문에 버전 관리 및 결과물에 대한 기록작성에 대한 중요성을 많이 느꼈다.


