## ДЗ по Automatic Speech Recognition

Пайплайн обучения ASR-модели с логированием, сохранением чекпоинтов (в т.ч. и в Google Drive) и вычислением метрик.
Далее описана процедура запуска предобученной в этом пайплайне модели на произвольных данных.

## Структура репозитория
- `hw_asr`
1. `hw_asr/configs`: файлы конфигурации

файл конфигурации (JSON) полностью определяет процесс обучения (какая модель берётся, какие данные используютс, как и какие логируются метрики)

2. `hw_asr/tests`: тесты реализации

Можно запустить все командой
```
python3 -m unittest discover hw_asr/tests
```

3. остальные поддиректории: исходный код пайплайна

- `notebooks`: Jupyter-ноутбуки с демонстрацией какой-либо функциональности пайплайна (сейчас там только аугментации)

- `requirements.txt`: Python-пакеты, необходимые для запуска

#### Скрипты

Подробнее про каждый из них можно узнать, вызвав `python3 {script_name} --help`.

- `train.py`: запуск процедуры обучения из консоли

- `test.py`: запуск оценки сохранённой модели из консоли

- `model_loader.py`: импорт моделей из внешнего хранилища (реализовано только Google Drive)

#### Файлы для импорта моделей из Google Drive:

Используются в скрипте `model_loader.py`.

- `gdrive_storage/external_storage.json`: файл конфигурации внешнего хранилища
- `gdrive_storage/gdrive_models_storage_key.json`: ключ для доступа к Google Drive

## Подготовка к работе пайплайна

Работа пайплайна проверялась на версиях Python 3.8 и 3.10.

<u>Для использования пайплайна необходимо:</u>
1. Установить все необходимые пакеты:
```
pip install -r requirements.txt
```

<u>Для загрузки предобученной модели:</u>

1. Необходимо вызвать скрипт с указанием названия нужного запуска и его чекпоинта.

Переменные, используемые в скриптах далее:
- `EXP="kaggle_deepspeech2_1+6_finetuning"` - имя эксперимента
- `RUN="finetuned1"` - имя запуска
- `CHECKPOINT="model_best"` - имя чекпоинта

Представленные значения соответствуют предобученной модели.

```
python3 model_loader.py \
   --config=gdrive_storage/external_storage.json \
   --run=$EXP:$RUN \
   config
python3 model_loader.py \
   --config=gdrive_storage/external_storage.json \
   --run=$EXP:$RUN \
   checkpoint $CHECKPOINT
```

> При запуске скриптов из директории репозитория конфиг `gdrive_storage/external_storage.json` не требуется модифицировать, иначе - нужно обновить в нём путь до файла с ключом

Также с помощью аргумента `-p, --path` можно указать директорию сохранения конфига и чекпоинта (по умолчанию: `saved/models`).

В случае каких-либо проблем со стороны API Google Drive загрузить модель можно вручную по [ссылке](https://drive.google.com/drive/folders/1k7JkQV9ZBwQTKEYfJqt78gI5ko6NtYN-?usp=drive_link).


## Как обучалась предобученная модель

Вызовом команды

```
python3 train.py \
   --config=hw_asr/configs/1+6/kaggle_deepspeech2_1+6_bidir_gru.json
```

И затем (для дообучения) вызовом
```
python3 test.py \
   --config=hw_asr/configs/1+6/kaggle_deepspeech2_1+6_other_finetuning.json \
   --resume=saved/models/kaggle_deepspeech2_1+6/$RUN_NAME/model_best.pth
```
, где $RUN_NAME - имя запуска в первом случае

> Повторить это без дополнительных действий не получится, т.к. в конфиге указано сохранение чекпоинтов на Google Drive, для чего локально использовался отдельный файл с ключом и доп. авторизация в аккаунт.
Достаточно убрать из файла конфигурации запись с "external_storage", чтобы всё заработало без экспортирования чекпоинтов на Google Drive.

**TODO**: добавить инструкцию по экспорту чекпоинтов на Google Drive через `gdrive_storage`


## Как запустить предобученную модель

Для того, чтобы оценить модель на датасете, можно задать конфиг с датасетом по аналогии с конфигами из `hw_asr/configs/eval_metrics_configs`, в котором также можно указать декодер выходного текста (в конфигах-примерах это CTC-декодер с LM).

> В конфигах `test-clean.json` и `test-other.json` указаны параметры для языковой модели, которые были подобраны по датасету Librispeech dev-clean.
Предположительно с этими же параметрами результат будет лучше и на других данных.

Далее пример запуска скрипта `test.py`:
```
python3 test.py \
   --config=hw_asr/configs/eval_metrics_configs/test-clean.json \
   --resume=saved/models/$EXP/$RUN/$CHECKPOINT.pth
```

> Выбранные языковая модель и данные будут загружены при запуске скрипта.

- Аргумент `-c, --config`: путь до дополнительного конфига.
Основной возьмётся из той же директории, что и указанный чекпоинт.
Итоговый конфиг берётся как основной, объединённый с дополнительным и с полем `data` из дополнительного, если оно в нём определено.

- Аргумент `-r, --resume`: путь до чекпоинта модели, которую мы хотим запустить.

В результате в терминале напечаются значения метрик CER и WER при разных вариантах декодирования предсказанного текста по вероятностям из модели (argmax, beam search, beam search + LM).
Также в файл, переданный как аргумент `-o, --output` (по умолчанию - `"output.json"`) будут записаны предсказания модели вместе с верными ответами из датасета.

Вместо создания отдельного конфигурационного файла для датасета можно в качестве аргумента (`-t`, `--test-data-folder`) указать путь до директории, имеющей следующую структуру:
```
test_dir
|-- audio
|    |-- voice1.[wav|mp3|flac|m4a]
|    |-- voice2.[wav|mp3|flac|m4a]
|-- transcriptions
|    |-- voice1.txt
|    |-- voice2.txt
```

## Автор

Егоров Егор:
- tg: [@TrickmanOff](https://t.me/TrickmanOff)
- e-mail: yegyegorov@gmail.com