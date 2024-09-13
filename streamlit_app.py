import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st 

from regression import Linear_Regression

t = np.linspace(1, 10, num=10)
views = np.array([5252, 7620, 941, 1159, 485, 299, 239, 195, 181, 180])
regs = np.array([21, 46, 9, 8, 3, 6, 4, 2, 2, 2])
lr = Linear_Regression(views, regs)
lr.fit()

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

col1, col2 = st.columns(2)

with col1:
    tab1, tab2, tab3 = st.tabs(["Среднее значение", "Дисперсия", "Корреляция"])

    with tab1:
        formula, python = st.tabs(["Формула", "Python"])
        formula.markdown("$m(x)=\\frac{1}{N}*\sum_{i=0}^{N-1} x_i$")
        python.code("""
            def m(X):
                return sum(X)/len(X)
        """)

    with tab2:
        formula, python = st.tabs(["Формула", "Python"])
        formula.markdown("$D(x)=\\frac{1}{N}*\sum_{i=0}^{N-1}(x_i-m(x))^2$")
        python.code("""
            def d(X):
                return m(X**2) - (m(X))**2
        """)

    with tab3:
        formula, python = st.tabs(["Формула", "Python"])
        formula.markdown("$\\frac{1}{N}*\\frac{\sum_{i=0}^{N-1}(x_i-m(x))(z_i-m(z))}{\sqrt{D(x)*D(z)}}$")
        python.code("""
            def correlation(X, Y):
                return sum((X - m(X)) * (Y - m(Y))) / (len(X) * (d(X) * d(Y)) ** 0.5)
        """)

with col2:
    st.code(
    """
    class Linear_Regression:
    def __init__(self, X, Y):
        self.x = X
        self.y = Y

    def m(self, X):
        return sum(X)/len(X)

    def d(self, X):
        return self.m(X**2) - (self.m(X))**2

    def correlation(self, X, Y):
        return sum((X - self.m(X)) * (Y - self.m(Y))) / (len(X) * (self.d(X) * self.d(Y)) ** 0.5)

    def fit(self):
        self.a = (self.m(self.y)*self.m(self.x)-(self.m(self.y*self.x)))/(self.m(self.x)**2-self.m(self.x**2))
        self.b = self.m(self.y)-self.a*self.m(self.x)

    def predict(self, X):
        return self.a*X + self.b
    """
    )

st.markdown("#### Сравнение работы :red[написанных] и :blue[встроенных] функций")

table = pd.DataFrame(
    [["M(x)", lr.m(views), np.mean(views), abs(lr.m(views) - np.mean(views))],
     ["D(x)", lr.d(views), np.var(views), abs(lr.d(views) - np.var(views))],
     ["Регрессия", lr.correlation(views, regs), np.corrcoef(views, regs)[0][1], abs(np.corrcoef(views, regs)[0][1] - lr.correlation(views, regs))]],
     columns=['Функция', 'Собственная', 'NumPy', 'Разница']
)
st.dataframe(table)

st.markdown(f"Коэффициент A: {lr.a}")
st.markdown(f"Свободный член B: {lr.b}")

fig = plt.figure(figsize=(10, 4))
plt.title("Конверсия пользователей")
plt.plot(views, lr.a*views+lr.b, label="Линейная регрессия")
plt.scatter(views, regs, color="red", label="Исходные данные")
plt.xlabel('views', size=13)
plt.ylabel('regs', size=13)
plt.legend()

st.pyplot(fig)

st.markdown("#### Предсказание количества просмотров:")
predict_number = st.slider("Количество просмотров: ", 20, 10000, 1000)
st.markdown(f"Количество регистраций: Y = {lr.predict(np.array([int(predict_number)]))[0]}")

fig2 = plt.figure(figsize=(10, 4))
plt.plot(views, lr.a*views+lr.b, label="Линейная регрессия")
plt.scatter(np.array([int(predict_number)]), lr.predict(np.array([int(predict_number)])), color="green", label="Предсказываемое количество регистраций")
plt.scatter(views, regs, color="red", label="Исходные данные")
plt.xlabel('views', size=13)
plt.ylabel('regs', size=13)
plt.legend()

st.pyplot(fig2)