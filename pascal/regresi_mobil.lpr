program regresi_mobil;

uses
  multiarray,
  numerik,
  gnuplot;

var
  DataMobil, x, y, Err: TMultiArray;
  w, b, dw, db, yPred: TMultiArray;
  JmlSampel, JmlFeature, i: integer;
  lr: single;
  fig: TFigure;


begin
  DataMobil := ReadCSV('../dataset/car_price.csv');

  x := DataMobil[[ _ALL_ , Range(0, 13)]];
  x := (x - Mean(x, 0)) / Std(x, 0);

  JmlSampel := x.Shape[0];
  JmlFeature := x.Shape[1];

  y := DataMobil[[_ALL_, 13]];

  w := Zeros([JmlFeature, 1]);
  b := 0;

  lr := 0.001;
  for i := 1 to 2000 do
  begin
    yPred := Matmul(x, w) + b;

    dW := (1 / JmlSampel) * Matmul(x.T, (yPred - y));
    db := (1 / JmlSampel) * Sum(yPred - y);

    w := w - lr * dw;
    b := b - lr * db;

    Err := Mean((yPred - y) ** 2); // hitung error
    PrintMultiArray(Err); // cetak error
  end;
  WriteCSV(yPred, 'prediction.csv'); // simpan hasil prediksi

  //fig := TFigure.Create('Prediksi harga mobil', 'id mobil', 'Harga');
  //fig.AddLinePlot(y, 'sebenarnya');
  //fig.AddLinePlot(yPred, 'prediksi');
  //fig.Show;

  ReadLn;
end.
