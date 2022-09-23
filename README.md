# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev]
Отчет по лабораторной работе #1 выполнил:
- Батраков Дмитрий Антонович
- НМТ212701
Отметка о выполнении заданий (заполняется студентом):

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 60 |
| Задание 2 | * | 20 |
| Задание 3 | # | 20 |

знак "*" - задание выполнено; знак "#" - задание не выполнено;

Работу проверили:
- к.т.н., доцент Денисов Д.В.
- к.э.н., доцент Панов М.А.
- ст. преп., Фадеев В.О.

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Структура отчета

- Данные о работе: название работы, фио, группа, выполненные задания.
- Цель работы.
- Задание 1.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 2.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 3.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Выводы.
- ✨Magic ✨

## Цель работы
Ознакомиться с основными операторами зыка Python на примере реализации линейной регрессии.

## Задание 1
### Написать программу Hello Word на Python и  Unity

код на Python
```py
print("Hello Word")
```
Вывод программы 
```py
Hello Word
```
код на Unity (C#)
...

## Задание 2
### Пошагово выполнить каждый пункт раздела "ход работы" с описанием и примерами реализации задач

- Подготовка данных для работы с алгоритмом линейной регрессии. 10 видов данных были установлены случайным образом, и данные находились в линейной зависимости. Данные преобразуются в формат массива, чтобы их можно было вычислить напрямую при использовании умножения и сложения.

```py
In [ ]:
#Import the required modules, numpy for calculation, and Matplotlib for drawing
import numpy as np
import matplotlib.pyplot as plt
#This code is for jupyter Notebook only
%matplotlib inline

# define data, and change list to array
x = [3,21,22,34,54,34,55,67,89,99]
x = np.array(x)
y = [2,22,24,65,79,82,55,130,150,199]
y = np.array(y)

#Show the effect of a scatter plot
plt.scatter(x,y)
```
- Определеие связанных функций. Функция модели: определяет модель линейной регрессии wx+b. Функция потерь: функция потерь среднеквадратичной ошибки. Функция оптимизации: метод градиентного спуска для нахождения частных производных w и b.

```py
#In [ ]:
#The basic linear regression model i wx+ b, and since this is a two-dimensional spase, the model is ax+ b

def model (a, b, x):
  return a*x + b

#The most commonly used loss function of liner regression model is the loss function of mean variance difference
def loss_functional (a, b, x, y):
  num = len(x)
  prediction = model (a, b, x)
  return (0.5/num)*(np.square(prediction)).sum()

#The optimization function mainly USES partial derivaties to update two parameters a and b
def optimize (a, b, x, y):
  num = len(x)
  prediction = model (a, b, x)
  #Update the values of A and B by finding the partial derivaties of loss the function on a and b
  da = (1.0 / num) * ((prediction - y) * x).sum()
  db = (1.0 / num) * (prediction - y).sum()
  a = a - Lr * da
  b = b - Lr * db
  return a, b

#iterated function, return a and b
def iterate (a, b, x, y, times):
  for i in range(times):
    a,b = optimize(a, b, x, y)
  return a,b
```

Шаг 1: Инициализация и модель итеративноц оптимизации

```py
#Initialize parameters and display
a = np.random.rand(1)
print(a)
b = np.random.rand(1)
print(b)
Lr = 0.000001

#For the first iteration, the parameter values, losses, and visualization after the iteration are displayed
a, b = iterate(a, b, x, y, 1)
prediction = model(a, b, x)
loss = loss_function(a, b, x, y)
print(a, b, loss)
plt.scatter(x, y)
plt.plot(x, prediction)
```
[0.87434118]
[0.02076263]
[0.87709954] [0.02080162] 1208.602044837438
[<matplotlib.lines.Line2D at 0x7fcf7e90c310>]

![image](https://user-images.githubusercontent.com/113825126/191679343-3a0f32ee-dc3d-4b50-9892-cc99c067eb4a.png)

Шаг 2: На второй итерации отображаются значения параметров, значения потерь  иэффекты визуллизации после итерации

```py
a, b = iterate(a, b, x, y, 2)
prediction = model(a, b, x)
loss = loss_function(a, b, x, y)
print(a, b, loss)
plt.scatter(x, y)
plt.plot(x, prediction)
```
[0.38689728]
[0.9097233]
[0.39537661] [0.90984589] 263.02004152619264
[<matplotlib.lines.Line2D at 0x7fcf7e96a790>]

![image](https://user-images.githubusercontent.com/113825126/191682420-a259bff6-a158-47b5-bcb9-2105c2d13495.png)

Шаг 3: Третья итерация показывает значения параметров, значения потерь и визуализацию после итерации

```py
a, b = iterate(a, b, x, y, 3)
prediction = model(a, b, x)
loss = loss_function(a, b, x, y)
print(a, b, loss)
plt.scatter(x, y)
plt.plot(x, prediction)
```
[0.84765244]
[0.13306453]
[0.85613608] [0.13318457] 1156.1472379508996
[<matplotlib.lines.Line2D at 0x7fcf7e8561d0>]

![image](https://user-images.githubusercontent.com/113825126/191682744-d3429610-25e2-46ef-b107-a727eef0c2ff.png)

Шаг 4: На четвёртой итерации отображаются значения праметров, значения потерь и эффекты визуализации

```py
a, b = iterate(a, b, x, y, 4)
prediction = model(a, b, x)
loss = loss_function(a, b, x, y)
print(a, b, loss)
plt.scatter(x, y)
plt.plot(x, prediction)
```
[0.52954468]
[0.82661341]
[0.54468287] [0.82683097] 487.6261009104903
[<matplotlib.lines.Line2D at 0x7fcf7e7d7bd0>]

![image](https://user-images.githubusercontent.com/113825126/191682923-ffafc4e1-0f29-40fc-a96a-557141d305a8.png)

Шаг 5: Пятая итерация показывает значения параметра, значени потерь, значени потерь и эффект визуализации после итерации

```py
a, b = iterate(a, b, x, y, 5)
prediction = model(a, b, x)
loss = loss_function(a, b, x, y)
print(a, b, loss)
plt.scatter(x, y)
plt.plot(x, prediction)
```
[0.02381831]
[0.22918067]
[0.05074296] [0.22957525] 4.625444773234256
[<matplotlib.lines.Line2D at 0x7fcf7e755710>]

![image](https://user-images.githubusercontent.com/113825126/191683245-0d0f4c76-7a54-401a-b492-b30d24f73e56.png)

Шаг 6: 1000-я итерация, показывает значение параметра, значение ротерь и эффект визуализации после итерации

```py
a, b = iterate(a, b, x, y, 5)
prediction = model(a, b, x)
loss = loss_function(a, b, x, y)
print(a, b, loss)
plt.scatter(x, y)
plt.plot(x, prediction)
```
[0.10296499]
[0.17649516]
[1.67922886] [0.19743818] 4442.686299950896
[<matplotlib.lines.Line2D at 0x7fcf7e6ce110>]

![image](https://user-images.githubusercontent.com/113825126/191683363-31ead9ed-5ad1-48c3-90a2-8b5df650a74b.png)

## Задание 3
### Изучить код на Python и ответить на вопросы:
-Должна ли величина Loss стремиться к нулю при исходных данных? Ответить на вопросю, привести пример выполнения кода, который подтверждает ответ.

Ответ: величина Loss вычисляется с использованием величин a и b, которые определяются случайно, поэтому каждый раз при компиляции Loss имеет разные значения (около 20, 70, 450, 1970 и другие). Исходя из названия и метода вычисления величины Loss, она должна стремиться к нулю, но на практике этого не наблюдается.

-Какова роль параметра Lr? Ответить на вопрос, привисти пример выполнения кода, который подтверждает ответ. В качестве эксперемента разрешается изменить параметр.
Ответ: при увеличении параметра Lr (последовательное увеличение параметра в 10 раз до значения 10), увеличился угол между графиком и осью абцисс, также увеличивались значения на оси ординат, линия графика удалялась от выделенных точек. Параметр влияет на определение регресси.
## Выводы

Абзац умных слов о том, что было сделано и что было узнано.

...

## Powered by

**BigDigital Team: Denisov | Fadeev | Panov**
