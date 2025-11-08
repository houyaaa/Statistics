# 통계학 6주차 정규과제

📌통계학 정규과제는 매주 정해진 분량의 『*데이터 분석가가 반드시 알아야 할 모든 것*』 을 읽고 학습하는 것입니다. 이번 주는 아래의 **Statistics_6th_TIL**에 나열된 분량을 읽고 `학습 목표`에 맞게 공부하시면 됩니다.

아래의 문제를 풀어보며 학습 내용을 점검하세요. 문제를 해결하는 과정에서 개념을 스스로 정리하고, 필요한 경우 추가자료와 교재를 다시 참고하여 보완하는 것이 좋습니다.

6주차는 `2부-데이터 분석 준비하기`를 읽고 새롭게 배운 내용을 정리해주시면 됩니다


## Statistics_6th_TIL

### 2부. 데이터 분석 준비하기

### 11. 데이터 전처리와 파생변수 생성

<!-- 11. 데이터 전처리와 파생변수 생성에서 11.1 결측값 처리부터 11.4 데이터 표준화와 정규화 스케일링 파트까지 진행해주시면 됩니다. -->

## Study ScheduleStudy Schedule

| 주차  | 공부 범위     | 완료 여부 |
| ----- | ------------- | --------- |
| 1주차 | 1부 p.2~46    | ✅         |
| 2주차 | 1부 p.47~81   | ✅         |
| 3주차 | 2부 p.82~120  | ✅         |
| 4주차 | 2부 p.121~167 | ✅         |
| 5주차 | 2부 p.168~202 | ✅         |
| 6주차 | 2부 p.203~250 | ✅         |
| 7주차 | 2부 p.251~299 | 🍽️         |

> 과제가 많이 남지 않았습니다. 조금만 더 화이팅해주세요!

<!-- 여기까진 그대로 둬 주세요-->



---

# 1️⃣ 개념 정리 

## 11.데이터 전처리와 파생변수 생성

```
✅ 학습 목표 :
* 결측값과 이상치를 식별하고 적절한 방법으로 처리할 수 있다.
* 데이터 변환과 가공 기법을 학습하고 활용할 수 있다.
* 모델 성능 향상을 위한 파생 변수를 생성하고 활용할 수 있다.
```

### 11.1. 결측값 처리

**결측치 발생 종류**

1. 완전 무작위 결측
    - 결측값이 무작위로 발생한 경우 
    - 결측치가 나타난 데이터 자체를 제거 
2. 무작위 결측
    - 다른 변수의 특성에 의해 해당 변수의 결측치가 체계적으로 발생한 경우
    - ex) A마트의 전국 체인 매출 정보 중, 특정 체인점의 오류로 매출 정보에 결측값이 많이 나타난 경우 

3. 비무작위 결측
    - 결측값들이 해당 변수 자체의 특성을 갖고 있는 경우 
    - ex) A마트 고객정보 데이터에서 개인정보 공개를 선호하지 않아서 결측이 발생한 경우


**결측값 처리 방법**

1. 표본 제거 방법
    - 가장 간단한 방법
    - 결측값이 심하게 많은 변수 제거
    - 전체 데이터에서 결측값 비율 10%미만

2. 평균 대치법
    - 결측값을 제외한 온전한 값들의 평균을 결측값들에 대치
    - 장점: 사용 간단 + 결측 표본 제거 방법의 단점 보완
    - 단점: 통계량의 표준오차가 축소되어 왜곡됨

3. 보간법
    - 시계열적 특성을 가지고 있는 데이터 
    - ex) 매출 데이터의 일별 판매금액 변수의 결측값을 대치하고자 하는 경우 -> 전 시점 or 다음 시점 값, 평균값으로 대치 가능 

4. 회귀대치법
    - 해당 변수와 다른 변수 사이의 관계성을 고려하여 결측값 계산
    - ex) 연령 변수의 결측값 -> 연 수입 변수 사용 
    - 추정하고자 하는 결측값을 가진 변수: 종속변수 / 나머지 변수: 독립변수 
    - 결측된 변수의 분산을 과소 추정하는 단점
    -> 4-1. 확률적 회귀대치법
        - 인위적으로 회귀식에 확률 오차항을 추가
        - 관측된 값들을 변동성만큼 결측값에도 같은 변동성 추가 

5. 다중 대치법
    - 단순대치 여러 번 수행 -> n개의 가상적 데이터 생성 이들의 평균으로 결측값을 대치하는 방법

     **1단계 대치단계**: 가능한 대치 값의 분포에서 추출된 서로 다른 값으로 결측치를 처리한 n개의 데이터셋 생성

     **2단계 분석단계**: 생성된 각각의 데이터셋을 분석하여 모수의 추정치와 표준오차 계산

     **3단계 결합단계**: 계산된 각 데이터셋의 추정치와 표준오차를 결합하여 최종 결측 대치값 산출


#### **결측값 처리 실습**

**1. 빈 문자열 결측값**

 NaN, None이 아닌 빈칸(빈 문자열인 값)으로 되어있는 결측값이 있을 수 있음

```
def is_emptystring(x):
  return x.eq(" ").any()

df.apply(lambda x: is_emptystring(x))
```

**2. dropna()함수**

```
#모든 컬럼이 결측값인 행 제거
df_drop_all=df.dropna(how="all")

#세 개 이상의 컬럼이 결측값인 행 제거
df_drop_3=df.dropna(thresh=3)

#특정 컬럼(temp)가 결측값인 행 제거
df_drop_slt=df.dropna(subset=["temp"])

#한 컬럼이라도 결측치가 있는 행 제거 
df_drop_any=df.dropna(how="any")
```

**3. fillna(), 대치법**

결측치에 원하는 값을 채워 넣는 함수 

- DataFrame.fillna(value, inplace=False)
value: 결측치를 대체할 값 (숫자, 문자열, 평균값 등)

inplace: True로 설정하면 원본 데이터를 수정, False면 새로운 데이터프레임 반환

**+) inplace=False,True 차이**

df_filled = df.fillna(0): df_filled라는 새로운 데이터프레임을 생성

df.fillna(0, inplace=True): 기존 df에 덮어씌움

- DataFrame.fillna({'col':value})
특정 컬럼만 개별적으로 채울 때 활용 (inplace=False가 기본값)

```
## 결측값 기본 대치 방법들

# 특정값(0)으로 대치 - 전체 칼럼
df_0_all = df.fillna(0)

# 특정값(0)으로 대치 - 칼럼 지정
df_0_slt = df.fillna({'temp':0})

# 평균값 대치 - 전체 칼럼
df_mean_all = df.fillna(df.mean())

# 평균값 대치 - 칼럼 지정
df_mean_slt = df.fillna({'temp':df['temp'].mean()})

# 중앙값 대치 - 전체 칼럼
df_median_all = df.fillna(df.median())

# 중앙값 대치 - 칼럼 지정
df_median_slt = df.fillna({'temp':df['temp'].median()})

# 최빈값 대치 - 전체 칼럼
df_mode_all = df.fillna(df.mode())

# 최빈값 대치 - 칼럼 지정
df_mode_slt = df.fillna({'temp':df['temp'].mode()})

# 최대값 대치 - 전체 칼럼
df_max_all = df.fillna(df.max())

# 최대값 대치 - 칼럼 지정
df_max_slt = df.fillna({'temp':df['temp'].max()})

# 최소값 대치 - 전체 칼럼
df_min_all = df.fillna(df.min())

# 최소값 대치 - 칼럼 지정
df_min_slt = df.fillna({'temp':df['temp'].min(), 'hum':df['hum'].min()})
```

**4. fillna(), 보간법**

```
# 전 시점 값으로 대치 - 칼럼 지정
df1 = df.copy()
df1['temp'].fillna(method='pad', inplace=True)

# 뒤 시점 값으로 대치 - 전체 칼럼
df.fillna(method='bfill')

# 뒤 시점 값으로 대치 - 결측값 연속 한 번만 대치
df.fillna(method='bfill', limit=1)

# 보간법 함수를 사용하여 대치 - 단순 순서 방식
ts_intp_linear = df.interpolate(method='values')

# 보간법 함수를 사용하여 대치 - 시점 인덱스 사용

# dteday 칼럼 시계열 객체 변환
df['dteday'] = pd.to_datetime(df['dteday'])

# dteday 칼럼 인덱스 변경
df_i = df.set_index('dteday')

# 시점에 따른 보간법 적용
df_time = df_i.interpolate(method='time')

```

전 후 시점의 값과 동일한 값으로 대치 -> method='pad'or'bfill'

**5. 다중 대치법**

```
# 다중 대치(MICE)

# dteday 칼럼 제거
df_dp = df.drop(['dteday'], axis=1)

# 다중 대치 알고리즘 설정
imputer = IterativeImputer(imputation_order='ascending',
                           max_iter=10, random_state=42,
                           n_nearest_features=5)

# 다중 대치 적용
df_imputed = imputer.fit_transform(df_dp)

# 판다스 변환 및 칼럼 설정
df_imputed = pd.DataFrame(df_imputed)

df_imputed.columns = ['instant','season','yr','mnth','holiday','weekday','workingday','weathersit','temp','atemp','hum','windspeed','casual','registered','cnt']

```
1. dteday 칼럼 제거

IterativeImputer은 숫자형 데이터만 처리 가능 날짜형 컬럼 제거

2. IterativeImputer 설정

3. 다중 대치 적용

fit_transform()은 결측치를 예측하여 채운 numpy배열을 반환 
아직 dataframe형태가 아님 나중에 pd.dataframe()으로 다시 변환

4. pandas dataframe으로 변환

IterativeImputer는 컬럼명을 잃어버림
수동으로 원래 컬럼명 다시 지정


### 11.2. 이상치 처리

- 이상치: 일부관측치의 값이 전체 데이터의 범위에서 크게 벗어난 값
- 문제점: 분석 및 모델링의 정확도를 감소시킴
- 전체 데이터의 양이 많을수록 튀는 값이 통곗값에 미치는 영향력이 줄어듦 -> 이상치 제거 필요성 낮음 

*처리 방법*
1. 관측값 변경
    하한 값, 상한 값 결정 -> 하한 값, 상한 값으로 대체 

2. 가중치 조정
    이상치의 영향을 감소시키는 가중치를 주는 방법

*이상치 확인 방법*
- EDA, 데이터 시각화 
- 박스플롯 상에서 분류된 극단치 선정 

- 평균으로부터 +-표준편차 이상 떨어져있는 값
- 평균은 이상치에 민감하게 변함 -> 이상치에 보다 강건한 중위수와 중위수 절대 편차를 사용하는 것이 효과적 


*이상치에 대한 관점*

- 무조건적 이상치 탐색 지양
- 데이터 변수들의 의미, 비즈니스 도메인 이해 -> 이상치의 원인 파악
- ex) 경력,연봉 관계에서 전문직, 사무직 등으로 나눠서 분석할 수도 있음
- 도메인에 따라 중요한 분석요인 -> 제조 공정의 불량 원인 

#### 이상치 처리 실습

**1. describe()**

```
df['col'].describe()
```
컬럼의 min max mean 25% 값 등을 알 수 있음 

이상치 판단 기준 = IQR 3 (약 5시그마)

**2. boxplot**

```
sns.boxplot(y='col',data=dataframe)
plt.show()
```



### 11.3. 변수 구간화


- 목적: 데이터 분석 성능 향상, 해석의 편리성
- how? 이산형 변수 -> 범주형 변수 
- ex) 나이 -> 10대, 20대, 30대 등으로 범주형으로

*구간을 나누는 방법*
1. 클러스터링: 타깃 변수 설정 필요x, 구간화할 변수의 값들을 유사한 수준끼리 묶음
2. 의사결정나무: 타깃 변수 설정 후 구간화할 변수의 값을 타깃 변수 예측에 가장 적합한 구간으로 나누어 줌 

*변숫값이 효과적으로 구간화 되었는지 측정 지표*
1. WOE
2. IV: 수치가 높을수록 종속변수의 true와 false를 잘 구분할 수 있는 정보량이 많다는 의미 (변수가 종속변수를 제대로 설명할 수 있도록 구간화가 잘되면 IV값 높아짐)


#### 변수 구간화 실습

**1. df.insert()**

```
df.insert(loc, column, value)
df.insert(2,'BMI_bin',0)
```

loc위치,column제목,value무엇을 넣을건데

3번째 줄에 BMI_bin이라는 제목을 만들거고 그 아래 0을 채울 것임

**2. df.cut()**

```
pd.cut(x, bins, labels=None, right=True, include_lowest=False)
```

x	구간화할 데이터 (Series, list 등)
bins	구간의 경곗값 리스트 또는 구간 개수(int)
labels	각 구간의 이름 (범주형 레이블)
right	True면 오른쪽 경계 포함 (a, b], False면 왼쪽 포함 [a, b)
include_lowest	첫 번째 구간의 왼쪽 경계 포함 여부 (True면 포함)

### 11.4. 데이터 표준화와 정규화 스케일링

<!-- 새롭게 배운 내용을 자유롭게 정리해주세요. -->



<br>
<br>

---

# 2️⃣ 확인 과제

> **교재에 있는 실습 파트를 직접 따라 해보세요. 실습을 완료한 뒤, 결과화면(캡처 또는 코드 결과)을 첨부하여 인증해 주세요.**
>
> **단순 이론 암기보다, 직접 손으로 따라해보면서 실습해 보는 것이 가장 확실한 학습 방법입니다.**
>
> > **인증 예시 : 통계 프로그램 결과, 시각화 이미지 캡처 등**

![alt text](image-31.png)

~~~
인증 이미지가 없으면 과제 수행으로 인정되지 않습니다.
~~~



### 🎉 수고하셨습니다.