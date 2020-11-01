program klasifikasi_kanker;

uses
  multiarray,
  numerik,
  gnuplot;

var
  DataMobil, x, y, err: TMultiArray;
  w, b, dw, db, yPred: TMultiArray;
  JmlSampel, JmlFeature, i: integer;
  lr: single;
  fig: TFigure;

function Logistic(Pred: TMultiArray): TMultiArray;
begin
  Result := 1 / (1 + Exp(-(Pred)));
end;

begin
  DataMobil := ReadCSV('../dataset/cancer.csv');

  x := DataMobil[[_ALL_, Range(0, 30)]];
  x := (x - Mean(x, 0)) / Std(x, 0);

  JmlSampel := x.Shape[0];
  JmlFeature := x.Shape[1];

  y := DataMobil[[_ALL_, 30]];

  w := Zeros([JmlFeature, 1]);
  b := 0;

  lr := 0.001;
  for i := 1 to 2000 do
  begin
    yPred := Logistic(Matmul(x, w) + b);

    dW := (1.0 / JmlSampel) * Matmul(x.T, (yPred - y));
    db := (1.0 / JmlSampel) * Sum(yPred - y);

    err := -(1/JmlSampel) * sum(y * Ln(yPred) + (1 - y) * Ln(1 - yPred));
    WriteLn(err.Item);

    w := w - lr * dw;
    b := b - lr * db;
  end;

  fig := TFigure.Create('Prediksi kanker', '', 'prediksi');
  fig.AddScatterPlot(y, 'sebenarnya');
  fig.AddScatterPlot(yPred > 0.5, 'prediksi');
  fig.Show;

  ReadLn;
end.
