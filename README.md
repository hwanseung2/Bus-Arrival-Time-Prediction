# CLOVA AI RUSH 2022: Bus Arrival Time Prediction

<img src="https://img.shields.io/badge/Ubuntu-E95420?style=flat-square&logo=Ubuntu&logoColor=white"> <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"> <img src="https://img.shields.io/badge/Pytorch-EE4C2C?style=flat-square&logo=Pytorch&logoColor=white"> 

**4th** place solution to [CLOVA AI RUSH 2022: Bus arrival time prediction](https://campaign.naver.com/clova_airush/)


## ๐ฏ Result
RMSE error **46.36 sec** / baseline RMSE error 75.51 sec

## ๐  Dataset features
์ค์๊ฐ ๋ก๊ทธ๋ก ๊ตฌ์ฑ๋ ๋ฐ์ดํฐ๋ก, ๋ธ์ด์ฆ๊ฐ ์กด์ฌํ์ฌ ์ค์๊ฐ ๋ก๊ทธ ์ค k๋ฒ์งธ ํด๋น๋๋ ์ค์๊ฐ ๋ก๊ทธ๊ฐ ์กด์ฌํ์ง ์์ ์ ์๊ณ , ๋ฐ๋ผ์ ์ ๋ฅ์ฅ ์ํ์ค๊ฐ ์์ฐจ์ ์ด์ง ์์ ์ ์๋ ํน์ง์ด ์กด์ฌ.


## โ๏ธ Method
- ๋ฐ์ดํฐ๊ฐ ๋ฐ์ํ๋ ๊ณผ์ ๊ณผ ์์ง๋๋ ๊ณผ์ ์ ํ์ ํ, ์ํ์ค๊ฐ ์ค๋ณต๋๊ฑฐ๋ ์ ๊ฑฐ์ฅ ๊ฐ์ ์ด๋์๊ฐ์ด 5์ด ๋ฏธ๋ง์ด๊ฑฐ๋ ํน์  ์๊ฐ์ ์ด๊ณผํ๋ ๋ฑ์ ๋ธ์ด์ฆ ํ์

- ๋ฒ์ค ๋ก๊ทธ ๋ฐ์ดํฐ์ ๋ค๋ฅธ ์๊ณ์ด ๋ฐ์ดํฐ๊ฐ ๊ฐ์ง๋ ์ฐจ์ด์ ์ ํ์. ๊ธฐ์ ๊ณผ ์ข์  ์ฌ์ด๋ฅผ ์ดํํ๋ ์ฒ์๊ณผ ๋์ด ์ ํด์ ธ ์๊ณ , ๊ฐ์ ์ฃผํ ๋ธ์ ์์ ๋ค๋ฅธ ์์ผ, ๋ค๋ฅธ ์๊ฐ๋์ ์ดํ๋ ๊ธฐ๋ก ๋ฑ์ด ์กด์ฌํ๋ ์ ์ผ๋ก๋ถํฐ ๋ค๋ฅธ ์ฃผํ ๋ก๊ทธ๋ฅผ ์ด์ฉํ์ฌ ํ๊ท ๊ณผ ์ค์๊ฐ์ผ๋ก ์ ์ ํ imputation ์ํ

- ๋์ผํ ์ฃผํ ๋ธ์ ์ ๋ํด์ ๋ค๋ฅธ ์์ผ, ๋ค๋ฅธ ์๊ฐ์ด ๋ชจ๋ ๋ธ์ด์ฆ๋ก ์ ์ค๋ ๊ฒฝ์ฐ, ๋ฒ์ค๋ ํ์ฐจ ์ง์ ์ผ๋ก๋ถํฐ ๋ฐฉํฅ์ด ๋ฐ๋ ์ฑ๋ก ๋ค์ ๋ฐ๋ ๋ฐฉํฅ์ผ๋ก ์ดํํ๋ ์ ์ ํ์ฉ. ๋ฐ๋ํธ์์ ๊ฑธ๋ฆฐ ์์์๊ฐ์ ํ๊ท ์ผ๋ก missing value๋ฅผ ์ฒ๋ฆฌ.

- ํน์  ๋ฒ์ค์ $k$ ~ $k+1$ ๊ฑฐ๋ฆฌ๊ฐ ๋ชจ๋  ์ ๊ฑฐ์ฅ ๊ฐ์ ๊ฑฐ๋ฆฌ ์ค ์ ์ผ ๋์ ๊ฒฝ์ฐ, ํด๋น ๊ฑฐ๋ฆฌ ์์์๊ฐ๋ณด๋ค ์ค๋๊ฑธ๋ฆฌ๋ ์์์๊ฐ์  outlier๋ก ์ฒ๋ฆฌํ๊ณ  ํ๊ท ๊ณผ ์ค์๊ฐ์ ํ์ฉํ์ฌ ์ฒ๋ฆฌ.

- Informer๋ฅผ ํ์ฉํ์ฌ ๊ฐ๋ณ์ ์ธ ์ํ์ค ๊ธธ์ด์ ์์์๊ฐ๋ค์ ์์ธก.

## โญ Final solution
n-seed ensemble(bagging)


