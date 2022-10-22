# CLOVA AI RUSH 2022: Bus Arrival Time Prediction

<img src="https://img.shields.io/badge/Ubuntu-E95420?style=flat-square&logo=Ubuntu&logoColor=white"> <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"> <img src="https://img.shields.io/badge/Pytorch-EE4C2C?style=flat-square&logo=Pytorch&logoColor=white"> 

**4th** place solution to [CLOVA AI RUSH 2022: Bus arrival time prediction](https://campaign.naver.com/clova_airush/)


## 🎯 Result
RMSE error **46.36 sec** / baseline RMSE error 75.51 sec

## 📌  Dataset features
실시간 로그로 구성된 데이터로, 노이즈가 존재하여 실시간 로그 중 k번째 해당되는 실시간 로그가 존재하지 않을 수 있고, 따라서 정류장 시퀀스가 순차적이지 않을 수 있는 특징이 존재.


## ✔️ Method
- 데이터가 발생하는 과정과 수집되는 과정을 파악 후, 시퀀스가 중복되거나 정거장 간의 이동시간이 5초 미만이거나 특정 시간을 초과하는 등의 노이즈 파악

- 버스 로그 데이터와 다른 시계열 데이터가 가지는 차이점을 파악. 기점과 종점 사이를 운행하니 처음과 끝이 정해져 있고, 같은 주행 노선에서 다른 요일, 다른 시간대에 운행된 기록 등이 존재하는 점으로부터 다른 주행 로그를 이용하여 평균과 중앙값으로 적절히 imputation 수행

- 동일한 주행 노선에 대해서 다른 요일, 다른 시간이 모두 노이즈로 유실된 경우, 버스는 회차 지점으로부터 방향이 바뀐 채로 다시 반대 방향으로 운행하는 점을 활용. 반대편에서 걸린 소요시간의 평균으로 missing value를 처리.

- 특정 버스의 $k$ ~ $k+1$ 거리가 모든 정거장 간의 거리 중 제일 높은 경우, 해당 거리 소요시간보다 오래걸리는 소요시간은  outlier로 처리하고 평균과 중앙값을 활용하여 처리.

- Informer를 활용하여 가변적인 시퀀스 길이의 소요시간들을 예측.

## ⭐ Final solution
n-seed ensemble(bagging)


