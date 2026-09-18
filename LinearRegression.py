import numpy as np

class LinearRegressionCustom:
    """
    Собственная реализация линейной регрессии.
    Поддерживает 3 способа обучения (параметр method):
    - 'analytical' — точное аналитическое решение (нормальное уравнение)
    - 'gd'         — обычный (batch) градиентный спуск, использует ВСЕ объекты на каждом шаге
    - 'sgd'        — стохастический градиентный спуск, использует ОДИН случайный объект на шаге
    """

    def __init__(self, method='sgd', learning_rate=0.01, n_epochs=1000, random_state=21):
        """
        method        — какой способ обучения использовать
        learning_rate — скорость обучения (gamma), насколько сильно двигаем веса на каждом шаге
        n_epochs      — сколько раз "пройтись" по данным при обучении (для gd и sgd)
        random_state  — число для фиксации случайности (чтобы результат был воспроизводим)
        """
        self.method = method
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.weights = None  # сюда сохраним обученные веса (включая свободный член b)

    def _add_bias(self, X):
        """
        Добавляет к матрице признаков X столбец из единиц.
        Это нужно, чтобы свободный член b (bias) можно было считать
        как обычный вес при "признаке", который всегда равен 1.
        """
        ones_column = np.ones((X.shape[0], 1))  # столбец единиц высотой = количество объектов
        return np.hstack([ones_column, X])  # приклеиваем этот столбец слева к X

    def fit(self, X, y):
        """
        Обучение модели — находим оптимальные веса.
        X — матрица признаков (строки — объекты, столбцы — признаки)
        y — вектор истинных значений целевой переменной
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        X_b = self._add_bias(X)  # добавили столбец единиц для свободного члена
        n_objects, n_features = X_b.shape  # n_objects — сколько строк (квартир),
        # n_features — сколько столбцов (включая единичный)

        if self.method == 'analytical':
            self._fit_analytical(X_b, y)
        elif self.method == 'gd':
            self._fit_gradient_descent(X_b, y, n_objects, n_features)
        elif self.method == 'sgd':
            self._fit_sgd(X_b, y, n_objects, n_features)
        else:
            raise ValueError("method должен быть 'analytical', 'gd' или 'sgd'")

        return self

    def _fit_analytical(self, X, y):
        """
        Аналитическое решение — нормальное уравнение:
        w = (X^T X)^(-1) X^T y

        X_b.T       — транспонированная матрица (X^T)
        @           — символ умножения матриц в numpy
        np.linalg.pinv(...) — берёт псевдообратную матрицу (аналог деления, но для матриц)
        """

        X_centered = X - X.mean(axis=0)  # центрируем признаки
        y_centered = y - y.mean()  # центрируем target

        # lstsq решает систему через SVD — устойчивее, чем обращение (X^T X)
        self.weights = np.linalg.lstsq(X_centered, y_centered, rcond=None)[0]
        self.intercept_ = y.mean() - X.mean(axis=0) @ self.weights

    def _fit_gradient_descent(self, X_b, y, n_objects, n_features):
        """
        Обычный (batch) градиентный спуск.
        Отличие от SGD: на КАЖДОМ шаге мы используем ВСЕ объекты сразу,
        а не один случайный.
        """
        rng = np.random.RandomState(self.random_state)  # свой "изолированный" генератор случайности

        # Начинаем со случайных небольших весов
        self.weights = rng.randn(n_features) * 0.01

        for epoch in range(self.n_epochs):
            predictions = X_b @ self.weights  # предсказания модели для ВСЕХ объектов сразу
            errors = predictions - y  # ошибка для каждого объекта

            # Градиент функции потерь MSE по весам, усреднённый по всем объектам:
            # это математически то же самое, что d(MSE)/dw
            gradient = (2 / n_objects) * (X_b.T @ errors)

            # Шаг градиентного спуска: theta_new = theta_old - gamma * gradient
            self.weights = self.weights - self.learning_rate * gradient

    def _fit_sgd(self, X_b, y, n_objects, n_features):
        """
        Стохастический градиентный спуск (SGD).
        Отличие от обычного GD: на каждом шаге мы берём ОДИН случайный объект
        и считаем градиент только по нему — это быстрее, но "шумнее".
        Детерминированная версия. Используем np.random.RandomState(seed) — это
        отдельный "генератор случайности", привязанный именно к этому объекту модели.
        """
        rng = np.random.RandomState(self.random_state)  # свой "изолированный" генератор случайности

        # Инициализация весов теперь тоже идёт через rng, а не через глобальный np.random
        self.weights = rng.randn(n_features) * 0.01

        for epoch in range(self.n_epochs):
            # Перемешиваем порядок объектов на каждой эпохе —
            # это стандартная практика для SGD, чтобы модель не запоминала порядок данных
            indices = rng.permutation(n_objects)

            for i in indices:
                x_i = X_b[i]  # признаки ОДНОГО случайного объекта (вектор)
                y_i = y[i]  # истинное значение для этого объекта (число)

                prediction_i = x_i @ self.weights  # предсказание для этого одного объекта
                error_i = prediction_i - y_i  # ошибка для этого одного объекта

                # Градиент, посчитанный только по ОДНОМУ объекту (а не по всем сразу)
                gradient_i = 2 * error_i * x_i

                # Обновляем веса сразу после каждого объекта (а не после прохода по всем)
                self.weights = self.weights - self.learning_rate * gradient_i

    def predict(self, X):
        """
        Делаем предсказания для новых данных X, используя уже обученные веса.
        """
        X = np.array(X, dtype=float)
        X_b = self._add_bias(X)  # не забываем добавить столбец единиц, как при обучении
        return X_b @ self.weights

def r_squared(y_true, y_pred):
        """
        Вычисляет коэффициент детерминации R².
        y_true — вектор истинных значений целевой переменной
        y_pred — вектор предсказаний модели
        """
        y_true = np.array(y_true, dtype=float)
        y_pred = np.array(y_pred, dtype=float)

        # SS_res — сумма квадратов ошибок модели (насколько модель ошиблась)
        ss_res = np.sum((y_true - y_pred) ** 2)

        # y_mean — среднее истинное значение целевой переменной по всей выборке
        y_mean = np.mean(y_true)

        # SS_tot — сумма квадратов отклонений истинных значений от их среднего
        # (насколько сильно сами данные разбросаны, без учёта модели вообще)
        ss_tot = np.sum((y_true - y_mean) ** 2)

        # Итоговая формула R²
        r2 = 1 - (ss_res / ss_tot)
        return r2

def mean_absolute_error_custom(y_true, y_pred):
        """
        MAE — среднее значение абсолютной (по модулю) ошибки.
        Показывает, на сколько в среднем модель ошибается, в тех же единицах,
        что и целевая переменная (например, в рублях).
        """
        y_true = np.array(y_true, dtype=float)
        y_pred = np.array(y_pred, dtype=float)
        return np.mean(np.abs(y_true - y_pred))

def root_mean_squared_error_custom(y_true, y_pred):
        """
        RMSE — корень из среднего квадрата ошибки.
        В отличие от MAE, сильнее "штрафует" за большие ошибки (выбросы),
        так как ошибки возводятся в квадрат перед усреднением.
        """
        y_true = np.array(y_true, dtype=float)
        y_pred = np.array(y_pred, dtype=float)
        return np.sqrt(np.mean((y_true - y_pred) ** 2))


class RegularizedLinearRegression(LinearRegressionCustom):
    """
    Регуляризация.
    penalty  — тип регуляризации: 'none', 'l2' (Ridge), 'l1' (Lasso), 'elasticnet'
    alpha    — сила регуляризации (λ) (умножается на каждую формулу регуляризации по типу)
    l1_ratio — используется только для elasticnet: определяет, какая доля штрафа приходится на l1, а какая — на l2
    """

    def __init__(self, method='sgd', learning_rate=0.01, n_epochs=1000,
                 random_state=None, penalty='none', alpha=0.0, l1_ratio=0.5, grad_clip_norm=None):
        super().__init__(method=method, learning_rate=learning_rate,
                         n_epochs=n_epochs, random_state=random_state)
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.grad_clip_norm = grad_clip_norm

    def _regularization_gradient(self):
        """
        Считает градиент штрафа (регуляризации) по весам.
        Bias (weights[0]) не регуляризуем — поэтому обнуляем его вклад отдельно.
        """
        reg_grad = np.zeros_like(self.weights)

        if self.penalty == 'none':
            return reg_grad  # штрафа нет вообще — обычная линейная регрессия

        if self.penalty == 'l2':
            # Производная от w^2 по w равна 2w — это и есть "сила притяжения к нулю",
            # которая ПРОПОРЦИОНАЛЬНА самому весу (чем больше вес, тем сильнее толкаем к 0)
            reg_grad = 2 * self.alpha * self.weights

        elif self.penalty == 'l1':
            # Производная от |w| — это sign(w): +1, если w > 0, -1, если w < 0,
            # и это значение НЕ зависит от величины веса (постоянная "сила толчка" к нулю —
            # именно поэтому L1 способна обнулять веса полностью)
            reg_grad = self.alpha * np.sign(self.weights)

        elif self.penalty == 'elasticnet':
            l1_part = self.l1_ratio * np.sign(self.weights)
            l2_part = (1 - self.l1_ratio) * 2 * self.weights
            reg_grad = self.alpha * (l1_part + l2_part)

        reg_grad[0] = 0  # bias не регуляризуем — обнуляем его "штрафной" градиент
        return reg_grad

    def _fit_sgd(self, X_b, y, n_objects, n_features):
        rng = np.random.RandomState(self.random_state)
        self.weights = rng.randn(n_features) * 0.01

        for epoch in range(self.n_epochs):
            indices = rng.permutation(n_objects)

            for i in indices:
                x_i = X_b[i]
                y_i = y[i]

                prediction_i = x_i @ self.weights
                error_i = prediction_i - y_i
                gradient_i = 2 * error_i * x_i

                # Делим регуляризацию на n_objects — иначе за одну эпоху штраф
                # применяется n раз (по разу на объект), что искусственно многократно
                # усиливает регуляризацию по сравнению с batch-версией при том же alpha
                gradient_i = gradient_i + self._regularization_gradient() / n_objects

                # Защита от "взрыва" градиента: если длина вектора градиента
                # больше max_norm, сжимаем его до этой длины, сохраняя направление.
                # Не даёт единичному экстремальному шагу окончательно испортить веса.
                # Обрезка применяется, ТОЛЬКО если явно задан grad_clip_norm при создании модели.
                if self.grad_clip_norm is not None:
                    grad_norm = np.linalg.norm(gradient_i)
                    if grad_norm > self.grad_clip_norm:
                        gradient_i = gradient_i * (self.grad_clip_norm / grad_norm)

                self.weights = self.weights - self.learning_rate * gradient_i

    def _fit_gradient_descent(self, X_b, y, n_objects, n_features):
        rng = np.random.RandomState(self.random_state)
        self.weights = rng.randn(n_features) * 0.01

        for epoch in range(self.n_epochs):
            predictions = X_b @ self.weights
            errors = predictions - y
            gradient = (2 / n_objects) * (X_b.T @ errors)

            gradient = gradient + self._regularization_gradient()

            self.weights = self.weights - self.learning_rate * gradient

    def _fit_analytical(self, X, y):
        """
        Аналитическое решение существует только для Ridge (L2) —
        формула: w = (X^T X + λI)^(-1) X^T y, где I — единичная матрица.

        Для Lasso и ElasticNet аналитического решения не существует,
        поэтому для них здесь выбрасываем ошибку.
        """
        X_centered = X - X.mean(axis=0)
        y_centered = y - y.mean()
        n, p = X_centered.shape

        if self.penalty == 'l2':
            # Добавляем p дополнительных "виртуальных" строк — по одной на признак
            A = np.vstack([X_centered, np.sqrt(self.alpha) * np.eye(p)])
            b = np.concatenate([y_centered, np.zeros(p)])
        else:
            A, b = X_centered, y_centered

        self.weights = np.linalg.lstsq(A, b, rcond=None)[0]
        self.intercept_ = y.mean() - X.mean(axis=0) @ self.weights


class MinMaxScalerCustom:
    """
    Собственная реализация MinMaxScaler.
    Приводит каждый признак к диапазону [0, 1].
    """

    def __init__(self):
        self.min_ = None  # сюда сохраним x_min для каждого признака (посчитанные на train)
        self.max_ = None  # сюда сохраним x_max для каждого признака (посчитанные на train)

    def fit(self, X):
        """
        Запоминает min и max по каждому столбцу (признаку) обучающей выборки.
        axis=0 означает "считать вдоль столбцов", то есть отдельно для каждого признака,
        а не одно общее число для всей матрицы.
        """
        X = np.array(X, dtype=float)
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        return self

    def transform(self, X):
        """
        Применяет уже вычисленные на fit() min и max к данным X.
        Используется отдельно для train и для test — но min_/max_ всегда те,
        что были посчитаны на train (обучающей выборке).
        """
        X = np.array(X, dtype=float)

        # Защита от деления на ноль — если какой-то признак постоянный
        # (min == max, например, все квартиры имеют одинаковое число спален),
        # заменяем диапазон на 1, чтобы не получить деление на 0
        range_ = self.max_ - self.min_
        range_ = np.where(range_ == 0, 1, range_)

        return (X - self.min_) / range_

    def fit_transform(self, X):
        """Удобный метод — сразу и запомнить min/max, и применить преобразование."""
        self.fit(X)
        return self.transform(X)


class StandardScalerCustom:
    """
    Собственная реализация StandardScaler.
    Приводит каждый признак к среднему 0 и стандартному отклонению 1.
    """

    def __init__(self):
        self.mean_ = None  # μ для каждого признака (посчитанное на train)
        self.std_ = None  # σ для каждого признака (посчитанное на train)

    def fit(self, X):
        X = np.array(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        return self

    def transform(self, X):
        X = np.array(X, dtype=float)

        # Та же защита от деления на ноль, что и в MinMaxScaler —
        # если признак постоянный, его стандартное отклонение равно 0
        std_safe = np.where(self.std_ == 0, 1, self.std_)

        return (X - self.mean_) / std_safe

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class MyPolynomialRegression:
    """
    Собственная реализация полиномиальной регрессии с поддержкой регуляризации.
    Использует численно устойчивые методы:
    - Для Linear/Ridge (без L1): SVD-разложение через np.linalg.lstsq
      (устойчиво к мультиколлинеарности, в отличие от прямого обращения X^T X)
    - Для Lasso/ElasticNet (с L1): координатный спуск
      (стандартный метод для L1, гораздо стабильнее SGD на полиномиальных признаках)
    """

    def __init__(self, penalty=None, alpha=0.0, l1_ratio=0.5,
                 max_iter=100000, tol=1e-4):
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None
        self.mean_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # Центрируем признаки и target — так свободный член можно
        # вынести за скобки и считать отдельно
        self.mean_ = X.mean(axis=0)
        A = X - self.mean_
        b = y - y.mean()
        n, p = A.shape

        if self.penalty in (None, 'l2'):
            # Линейная регрессия или Ridge — решаем через SVD
            if self.penalty == 'l2':
                # Для Ridge добавляем "виртуальные" строки sqrt(alpha)*I
                A = np.vstack([A, np.sqrt(self.alpha) * np.eye(p)])
                b = np.concatenate([b, np.zeros(p)])

            # lstsq решает через SVD — численно устойчиво к мультиколлинеарности
            self.coef_ = np.linalg.lstsq(A, b, rcond=None)[0]

        else:
            # Lasso или ElasticNet — координатный спуск
            ratio = 1.0 if self.penalty == 'l1' else self.l1_ratio
            l1 = self.alpha * ratio
            l2 = self.alpha * (1 - ratio)

            # Предвычисляем матрицу Грама и правую часть — это ускоряет спуск
            gram = A.T @ A / n
            rhs = A.T @ b / n
            self.coef_ = np.zeros(p)

            for iteration in range(self.max_iter):
                for j in range(p):
                    # Убираем вклад j-го признака из текущего предсказания
                    rho = rhs[j] - gram[j] @ self.coef_ + gram[j, j] * self.coef_[j]
                    denominator = gram[j, j] + l2

                    # Мягкое пороговое преобразование (soft thresholding) —
                    # именно оно обнуляет неважные коэффициенты при L1
                    if denominator != 0:
                        self.coef_[j] = (
                                np.sign(rho) * max(abs(rho) - l1, 0) / denominator
                        )
                    else:
                        self.coef_[j] = 0.0

                # Проверка сходимости: если градиент почти нулевой — выходим
                gradient = gram @ self.coef_ - rhs + l2 * self.coef_
                residual = np.where(
                    self.coef_ != 0,
                    abs(gradient + l1 * np.sign(self.coef_)),
                    np.maximum(abs(gradient) - l1, 0)
                )
                if residual.max() <= self.tol * max(1, abs(rhs).max()):
                    break
            else:
                import warnings
                warnings.warn(
                    'Координатный спуск не сошёлся за max_iter итераций. '
                    'Увеличьте max_iter.'
                )

        # Восстанавливаем свободный член из условия центрирования
        self.intercept_ = y.mean() - self.mean_ @ self.coef_
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_


