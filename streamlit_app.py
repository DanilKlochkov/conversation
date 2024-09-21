import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st 

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error


t = np.linspace(1, 10, num=10)
views = np.array([5252, 7620, 941, 1159, 485, 299, 239, 195, 181, 180])
regs = np.array([21, 46, 9, 8, 3, 6, 4, 2, 2, 2])

X3, y3 = make_regression(n_samples=14, n_features=1, noise=2, random_state=0)
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, random_state=0)

lr = LinearRegression()
lr.fit(X3_train, y3_train)

lasso = Lasso()
lasso.fit(X3_train, y3_train)

ridge = Ridge()
ridge.fit(X3_train, y3_train)

elnet = ElasticNet()
elnet.fit(X3_train, y3_train)

st.title("Конверсия пользователей")
st.markdown("""
На основе данных из статьи: https://habr.com/ru/companies/nerepetitor/articles/250633/

При поиске некой афинной функции $f(x) = ax + b$, чтобы ее график ближе всего сходился к точкам.
Сумма модулей отклонений приведет нас к линейной регрессии (стремится к минимуму): $\sum_{i}|f(x_i)-y_i|$

Необходимо вычислить частные производные и приравнять их к нулю.

$\\frac{\delta S}{\delta a}=-2\sum_{i=1}^N (y_i - a - bx_i), \\frac{\delta S}{\delta b}=-2\sum_{i=1}^N (y_i - a - bx_i)x_i$

Которые затем при раскрытии дадут нам коэфф. A и свободный член B.

$\\begin{aligned} & \hat{a}=\langle y\\rangle-\hat{b}\langle x\\rangle \\ & \hat{b}=\\frac{\langle x y \\rangle -\langle x\\rangle \langle y\\rangle}{\left\langle x^2\\right\\rangle-\langle x \\rangle^2}\end{aligned}$

""")


tab1, tab2, tab3 = st.tabs(["ElasticNet", "Lasso", "Ridge"])

with tab1:
    st.markdown("Линейная регрессия с L1 и L2 регуляризаторами")
    st.markdown("Минимизация целевой функции: $\\frac{1}{2N}*||Xw - y||^2_2 + \\alpha\\rho * ||w||_1 + \\frac{\\alpha(1-\\rho)}{2} * ||w||^2_2$")
    st.markdown("$\\alpha$ - коэффициэнт регуляризации, $\\rho$ - параметр управляющий соотношением между L1 и L2, $||w||_1$ - норма L1 (сумма абсолютных значений), $||w||^2_2$ - норма L2 (сумма квадратов)")
    
with tab2:
    st.markdown("Lasso - модель, оценивающая разреженные коэффициенты. Предпочитает решения с меньшим количеством ненулевых коэффициентов")
    st.markdown("Минимизация целевой функции: $\\frac{1}{2N}*||Xw - y||^2_2 + \\alpha * ||w||_1$")

with tab3:
    st.markdown("Ridge - модель, добавляет L2-регуляризацию к функции стоимости и больше всего снижает веса для признаков с высокой корреляцией")
    st.markdown("Минимизация целевой функции: $||Xw - y||^2_2 + \\alpha * ||w||^2_2$")

table = pd.DataFrame(
    [["MSE", mean_squared_error(y3_test, lasso.predict(X3_test)), mean_squared_error(y3_test, ridge.predict(X3_test)), mean_squared_error(y3_test, elnet.predict(X3_test))],
     ["MAE", mean_absolute_error(y3_test, lasso.predict(X3_test)), mean_absolute_error(y3_test, ridge.predict(X3_test)), mean_absolute_error(y3_test, elnet.predict(X3_test))]],
     columns=['Ошибка', 'Lasso', 'Ridge', 'ElasticNet']
)
st.dataframe(table)

fig2 = plt.figure(figsize=(10, 4))
plt.plot(X3, lr.predict(X3), label="Линейная регрессия")
plt.plot(X3, lasso.predict(X3), label="Лассо")
plt.plot(X3, ridge.predict(X3), label="Ridge")
plt.plot(X3, elnet.predict(X3), label="ElasticNet")
plt.scatter(X3, y3, color="red", label="Исходные данные")
plt.xlabel('X', size=13)
plt.ylabel('Y', size=13)
plt.legend()

st.pyplot(fig2)