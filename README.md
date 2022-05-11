# Gesture_recognition

Модель для определения жеста и реакции на него.

## Общее описание
Первая стадия - определение наличия лица на входящем видео потоке. Если лицо найдено, идет определение жеста на ряде следующих кадров. Далее в зависимости от жеста выполняется конкретное действие, все они будут описаны ниже.

***Крайне рекомендуется запускать из локального `Jupyter Notebook` из-за использования модуля `cv2.imshow`, напрямую не поддерживаемого в Colab***

**Особенности реализации**:
* за модель распознавания жестов из коробки взят аналог `ResNet18`;
* при использовании стандартной `MTCNN` и модели распознавания жеста из коробки при угадывании жеста без оптимизаций количество кадров в секунду приблизительно равно 5, если же распознавать лицо 1 раз в 10 кадров (как настроено сейчас), выстродействие вырастает более чем в 2 раза - в среднем 12 кадров в секунду на моем железе (дальнейшую оптимизацию можно делать, изменяя архитектуру модели распознавания жестов). Без фиксации лица модель может работать со скорость в ~120 кадров в секунду, но камера у меня с 30 ¯\\\_(ツ)\_/¯ ;
* распознавание жеста происходит не по 1 кадру, а по серии (настраиваемый параметр, по умолчанию 10), в рамках которой идет усреднение результата и выбора максимального. Такой подход позволяет избегать случайного фиксирования какого-либо жеста при перестроении руки (например, после показанного пальца вверх при убирании руки не будет фикироваться кулак, который скорее всего будет на мгновение показан);
* 
* сейчас стандартная модель обучена только на датасете моих жестов и в ограниченном пространстве, поэтому может иметь место некоторое переобучение, в дальнейшем требуется расширение датасета и дообучение модели распознавания жестов.

**Интерфейс и что вообще происходит**:
* в самом левом верхнем углу показано название жеста и уверенность модели;
* на жесты, требующие какого-либо действия, ниже названия жеста должна появиться строка с выполняемым действием (на текущий момент настроена задержка в 30 кадров для того, чтобы успеть отменить жест при ошибочной регистрации жеста);
* после показанного жеста модель должна реагировать иконкой такого же жеста в левом верхнем углу (при отсутствии жеста ничего не показывается);
* есть 2 типа активностей при жестах: *продолжительные* (нужно держать руку для постоянного выполнения действия) и *фиксированные* (после первого показа жеста начинается выполнение действия и дальнейшие эти же жесты игнорируются до смены жеста на другой). Обычно текст для продолжительных указывается во время выполнения самого действия, для фиксированного текст пишется до начала момента выполнения действия (во время вывода текста программа просто ждет);
* если были показаны подряд несколько жестов, программа продолжит корректно работать и запишет все действия в очередь, из которой они потом поступательно будут выполняться;
* предусмотрена отмена всех действия, попавших в очередь (жест по умолчанию - показанная боком правая рука ладонью вниз);
* можно сохранить видео ряд в формате `gif` установив `save_imgs` при объявлении объекта в `True`. Результат будет сохранен в `RESULTS_PATH` (по умолчанию `./results`, которая будет создана при ее отсутствии).