# Elevator_System_Using_DEVS_formalism

"시뮬레이션" 과목 팀 프로젝트의 일환으로 진행되었습니다.  
<DEVS 시뮬레이션을 사용한 엘레베이터 시뮬레이터 구현 및 효율적 운영 알고리즘 탐색> 입니다.  
1개월 동안 아이디어 구상 및 시뮬레이션 제작, 알고리즘 적용을 진행하였습니다.  

# Intro
실제 엘레베이터 운영 데이터를 구할 수 없는 상황 속 가장 효율적인 운영 알고리즘을 찾기 위해선 시뮬레이션을 구현 후 해당 환경에서 최적의 알고리즘을 찾아야 했습니다.  
때문에 확장 가능성을 염두해둔 환경을 구현하고 특정 상황으로 제한해 프로젝트를 진행합니다.  
이 프로젝트에선 8개 층, 3개의 엘레베이터를 가진 건물 환경을 가정합니다.  

# DEVS 형식 시뮬레이션 구현
<img width="650" height="360" alt="Image" src="https://github.com/user-attachments/assets/e4f5eefb-2776-43ad-81c7-95f0bdd87883" />
  
<img width="350" height="150" alt="Image" src="https://github.com/user-attachments/assets/ee032401-6148-4f99-83bf-013fb1ef5428" />
  
  
 $ Floor_i $ : 지수 분포를 따르는 랜덤 시간 경과 후 승객(현재층, 목적지층, 생성시간) 정보가 생성됩니다. TotalBuffer로 송신됩니다.  
   
<img width="350" height="165" alt="Image" src="https://github.com/user-attachments/assets/cb820547-794e-4809-a29f-49170c317c84" />    

 $ TotalBuffer $ : Floor를 통해 수신한 승객 정보를 각 층별, 방향(UP,DOWN)별 대기 큐(Global State)에 저장합니다.  
 $ Global\;State $ : Shared memory로서 DEVS 시스템 구현에 있어 용이하게 해줍니다. 하지만 이러한 구조는 DEVS 형식론에 있어 information leakage를 유발할 수 있어 유의해야합니다.  
  
<img width="365" height="215" alt="Image" src="https://github.com/user-attachments/assets/c3b62ed0-5b50-4f4e-8e26-370407ce01b3" />    

 $ Controller $ : $Elevator_i$ 객체의 다음 행동(Up,Down,Idle)을 지정해줍니다. $Global\;State$와 $Elevator$ 상태을 고려해 계산됩니다.  
  
<img width="350" height="250" alt="Image" src="https://github.com/user-attachments/assets/58742824-c109-4364-ac4f-c80e170c8dc8" />    

 $ Elevator_i $ : Controller의 명령을 받아 수행합니다. 수행 절차는 다음과 같습니다.   
 "승객 탑승 -> 명령 수행(1초 대기) -> 승객 하차 -> 명령 요청".  
  
<img width="350" height="110" alt="Image" src="https://github.com/user-attachments/assets/90c48116-90cc-41b6-9d98-63be7e71afaa" />    

 모든 객체는 Coupled Model-Building의 계층적 요소로 구성됩니다.  

# Baseline : ETA 기반 운영 알고리즘
ETA(Estimated Time of Arrival)는 도착 예정 시간을 뜻하며,  
해당 운영 알고리즘은 요청에 대한 각 호기 별 Score를 계산하여 최소값을 가진 호기가 해당 요청을 처리합니다.  
<img width="350" height="32" alt="Image" src="https://github.com/user-attachments/assets/e9f2481c-2265-4e50-bc8d-4050620365e1" />  
$WaitingTime$이 포함된 이유는 공정성을 위함이며 승객의 대기 시간이 늘어남에 따라 Score가 낮아지고 최우선 요청으로 처리될 수 있게 해줍니다.  

  1. ETA 계산 로직  
    a. 이미 탑승객이 있는 엘레베이터는 계속 운행하고 이동 중 같은 방향 요청을 처리합니다.  
    b. 엘레베이터가 비었을때 ( IDLE )  
       물리적 거리 계산 : $ETA=|Target_Floor-Current_Floor|$   
    c. 엘레베이터가 상승/하강 중일때  
        i. 같은 방향의 요청 - $ETA=|Target_Floor-Current_Floor|$  
       ii. 반대 방향의 요청 - $ETA=|Target_Floor-Current_Floor| +2$  
     
# 강화학습 모델 DDQN 도입
<img width="700" height="350" alt="Image" src="https://github.com/user-attachments/assets/e05767ab-7aff-4545-8c0f-7f2880dab6cd" />    
 $Controller$를 DDQN으로 바꿔넣는다.  
Input : 건물 전체의 층별 대기열 유무, 각 엘레베이터의 상태 ( 현재 층, 진행 방향, 탑승률, 탑승객 요청층 )  
Output : 3대에 대한 (UP,DOWN,IDLE) $3^3$가지의 제어 명령  

보상함수: 10*(직전에 운반한 승객 수 ) -  0.1 * (남아 있는 대기열) - 0.05 *(움직였다면)  

# 평가 Baseline, DDQN
<img width="350" height="235" alt="Image" src="https://github.com/user-attachments/assets/ab2ba87d-5eab-422a-8906-ba8036f85116" />  
<Baseline 시각화>  
<img width="350" height="235" alt="image" src="https://github.com/user-attachments/assets/ce42c246-42ef-4ed4-9c04-fa4f86158d99" />  
<DDQN 시각화>  

|운영 알고리즘|평균 승객 운반 (명)|평균 대기 시간 (초)|평균 탑승 시간 (초)|평균 운행 거리 (층)|  
|:------|:---:|:---:|:---:|:---:|  
|Baseline|464.5|3.06|2.17|1286.6|  
|DDQN|469.9|**2.85**|2.18|**1174.8**|  

분석  
평균 운행 거리에서 Baseline 대비 DDQN 알고리즘이 100 계층 정도를 덜 움직였고, 이것이 평균 대기 시간을 더 줄일 수 있게 하였다고 추정되었다.  

**왜 강화학습 보상함수를 평가지표로 지정하지 않았는가?**
