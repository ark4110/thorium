# Thorium 2 — веб-интерфейс

Минимальный интерфейс по образцу T.H.O.R.I.U.M. 1: File → Open / Upload model, 3D-окно модели
со свойствами элемента по щелчку, Analysis → Thermal analysis с ходом расчёта, просмотр результатов
(шаги, Animate! / Stop!, шкала температур, цвета по шагу или по всему решению, режимы отрисовки,
Decimate results, Save results as txt в формате первой версии). Расчёт выполняет программа
`thorium2` из `../thorium2`.

Next.js (TypeScript, Tailwind), 3D — three.js.

## Запуск

```
npm install
cp ../thorium2/build/thorium2 bin/thorium2      # или статическая сборка, см. ниже
echo "THORIUM_WEB_PASSWORD=<пароль>" > .env.local
npm run build
npx next start -p 3700 -H 127.0.0.1
```

Весь сайт закрыт паролем (HTTP Basic, имя пользователя любое): без `THORIUM_WEB_PASSWORD`
не пускает никого. Статическая сборка программы, не зависящая от библиотек системы:
`cmake -S ../thorium2 -B build-static -DTHORIUM2_TESTS=OFF -DTHORIUM2_OPENMP=OFF -DCMAKE_EXE_LINKER_FLAGS=-static`.

Данные — в `data/` (модели, запуски, результаты); примеры первой версии копируются туда из
`../thorium2/tests/data/v1_examples` при первом обращении. `data/`, `bin/` и `.env.local` в git не входят.

## Проверка

```
npx vitest run    # разбор и запись результатов, цветовая шкала, пароль, имена моделей
npx eslint src
```
