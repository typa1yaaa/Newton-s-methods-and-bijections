import math

# траппинговая частота (частота ловушки)
# omega_0 обычно лежит в диапазоне ~ 2π⋅(10 Гц−10 кГц)
OMEGA_0 = 2 * math.pi * 100  # пример: 100 Гц

def function(lambd, C):
    # считаем функцию f(lambd)
    # sqrt(lambd * (lambd - 1)) + ln(sqrt(lambd) + sqrt(lambd - 1)) = C
    # f(lambd) = sqrt(lambd * (lambd - 1)) + ln(sqrt(lambd) + sqrt(lambd - 1)) - C
    
    # ограничиваем область определения функции 
    if lambd < 1:
        return float('inf') 
    
    sqrt_lambd = math.sqrt(lambd) #sqrt(lambd)
    sqrt_lambd_minus_one = math.sqrt(lambd - 1) #sqrt(lambd - 1)
    term1 = sqrt_lambd * sqrt_lambd_minus_one # sqrt(lambd) * sqrt(lambd - 1)
    term2 = math.log(sqrt_lambd + sqrt_lambd_minus_one) # ln(sqrt(lambd) + sqrt(lambd - 1))
    return term1 + term2 - C # sqrt(lambd) * sqrt(lambd - 1) + ln(sqrt(lambd) + sqrt(lambd - 1)) - C


def d_function(lambd):
    # считаем производную от функции f(lambd) для метода Ньютона
    # ограничиваем область определения функции 
    if lambd <= 1:
        return float('inf')
    
    sqrt_lambd = math.sqrt(lambd) #sqrt(lambd)
    sqrt_lambd_minus_one = math.sqrt(lambd - 1) #sqrt(lambd - 1)
    term1 = (2*lambd - 1) / (2 * sqrt_lambd * sqrt_lambd_minus_one) # производная от sqrt(lambd * (lambd - 1))
    term2 = (1/(2 * sqrt_lambd) + 1/(2 * sqrt_lambd_minus_one)) / (sqrt_lambd + sqrt_lambd_minus_one) # производная от ln(sqrt(lambd) + sqrt(lambd - 1)
    return term1 + term2 # складываем части производных


def bisection_method(f, C, a, b, tol=1e-15, max_iter=5000):
    # метод бисекции
    print("\nметод бисекции:")
    print(f"начальный интервал: [{a}, {b}]")

    # проверка, что функция меняет знак на концах интервала
    if f(a, C) * f(b, C) >= 0:
        raise ValueError("функция должна иметь разные знаки на концах интервала")
    
    # проверка, не являются ли границы интервала а или b точным корнем 
    # если да, то возвращаем соответствующуу границу
    if f(a, C) == 0:
        return a
    if f(b, C) == 0:
        return b

    i = 0 # счетчик итераций 

    # основной цикл метода бисекции
    while b - a > tol and i < max_iter:
        # цикл выполняется, пока длина интервала больше заданной точности tol
        # и пока не достигнуто максимальное число итераций max_iter
        c = (a + b) / 2  # вычисление середины интервала
        fc = f(c, C)
        # если функция меняет знак на отрезке [a, c], корень находится там,
        # и мы сдвигаем правую границу b в точку c
        # иначе корень в [c, b], и мы сдвигаем левую границу a в точку c
        # проверка смены знака функции
        if f(a, C) * fc < 0:
            b = c  # корень в левой половине
        else:
            a = c  # корень в правой половине

        print(f"итерация {i+1}: λ = {c:.12f}, f(λ) = {fc:.3e}")
        i += 1 # увеличиваем счетчик итераций

    print(f"найдено решение за {i} итераций")
    return (a + b) / 2, i-1  # возврат среднего значения как приближённого корня


def newtons_method(f, df, C, x0, tol=1e-15, max_iter=5000):
    # метод Ньютона (с защитой от малой производной, защитой области определения)
    print("\nметод Ньютона:")
    print(f"начальное приближение: lambd_0 = {x0}")

    x_current = x0 # текущее приближение корня
    i = 0 # счетчик итераций
    converged = False # флаг сходимости
    
    # основной цикл метода Ньютона
    while i < max_iter and not converged:
        # цикл выполняется, пока флаг сходимости не равен True
        # и пока не достигнуто максимальное число итераций max_iter
        # вычисление функции и ее производной в текущей точке
        fx = f(x_current, C)
        dfx = df(x_current)

        # проверка нулевой производной
        if dfx == 0:
            return x_current
        
        
        dx = fx / dfx  # стандартный шаг Ньютона
        x_next = x_current - dx # вычисляем новое приближение

        # защита от выхода за область определения (lambd >= 1)
        if x_next < 1.0:
            x_next = 1.0 + abs(dx)/2 # корректируем положение
        
        print(f"итерация {i+1}: λ = {x_next:.12f}, f(λ) = {fx:.3e}")
        
        # проверка условия сходимости (достигнута ли требуемая точность)
        if abs(x_next - x_current) < tol:
            print(f"найдено решение за {i+1} итераций")
            converged = True
        
        # обновление переменных для следующей итерации
        x_current = x_next
        i += 1 # увеличиваем счетчик итераций

    return x_current, i # возвращаем последнее вычисленное значение


def auto_parameters(C):
    # опредение начальных параметров для методов бисекции и Ньютона
    # начальный интервал для бисекции: [1, C + 1] (т.к. f(1) = -C, f(C+1) > 0)
    a = 1.0
    b = max(2.0, C + 1.0)  # гарантируем, что b >= 2 (потому что при маленьком C может возрасти риск численных ошибок)
    x0 = (a + b) / 2 # начальное приближение для Ньютона - среднее между a и b
    return a, b, x0 


def solve_equation(C):
    # проведение эксперимента
    a, b, x0 = auto_parameters(C)
    
    # решение методом бисекции
    lambda_bisect, iter_bisect = bisection_method(function, C, a, b)

    # решение методом Ньютона
    lambda_newton, iter_newton = newtons_method(function, d_function, C, x0)
    
    print("\nсравнение результатов:")
    print(f"\nрешение уравнения для C = {C}")
    print(f"метод бисекции: lambd = {lambda_bisect:.12f}, iter = {iter_bisect}")
    print(f"метод Ньютона:  lambd = {lambda_newton:.12f}, iter = {iter_newton}")
    print(f"разница:        {abs(lambda_bisect - lambda_newton):.16e}")

def main():
    t = float(input("введите время t для расчета функции (значение float): "))
    if t <= 0:
        raise ValueError("t должно быть положительным числом")

    # рассчитываем коэф. C: 
    C = math.sqrt(2) * OMEGA_0 * t
    
    # проводим эксперимент
    solve_equation(C)

if __name__ == "__main__":
    main()