for i in $( seq 1 50)    # Цикл, i от 1 до 50
do
    python generate_data.py anonymous --file_count ${i} --njobs 5 --train_size 10000 --valid_size 4000    # Запуск скрипта generate_data.py с параметрами. Здесь ${i} - это текущее значение переменной i. Вероятно, создает искусственные данные для обучения модели
    if [ $((i%10)) -eq 0 ];then    # Если i делится на 10 без остатка
    python train.py anonymous  --exp_name 3_anonymous_dagger --file_count ${i} --epoch 100    # 100 эпох обучения
    else
    python train.py anonymous  --exp_name 3_anonymous_dagger --file_count ${i} --epoch 10    # 10 эпох
    fi
done
