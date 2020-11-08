program digit;

{$mode objfpc}{$H+}

uses
  numerik,
  multiarray,
  noe,
  noe.optimizer;

const
  MAX_ITER = 100;

var
  DatasetTrain, DatasetTest, X, XTest, Y, YTest, YPred, Hidden1, Hidden2: TTensor;
  p, W1, W2, W3, b1, b2, b3, loss: TTensor;
  ParamArr: array of TTensor;
  i: integer;

  ParamList: TTensorList;
  optim: TOptAdam;

begin
  DatasetTrain := ReadCSV('../dataset/optdigits-train.csv');
  X := DatasetTrain[[_ALL_, Range(0, 64)]] / 16;
  Y := BinarizeLabel(DatasetTrain[[_ALL_, 64]]);

  W1 := RandG(0, 1, [64, 100]) / 10;
  W2 := RandG(0, 1, [100, 20]) / 10;
  W3 := RandG(0, 1, [20, 10]) / 10;
  b1 := Zeros([100]);
  b2 := Zeros([20]);
  b3 := Zeros([10]);

  ParamList := TTensorList.Create();
  ParamArr := [W1, W2, W3, b1, b2, b3];
  for p in ParamArr do
  begin
    p.RequiresGrad := True;
    ParamList.Add(p);
  end;

  optim := TOptAdam.Create(ParamList);
  optim.LearningRate := 0.01;

  for i := 1 to MAX_ITER do
  begin
    Hidden1 := ReLU(X.Matmul(W1) + b1);
    Hidden2 := ReLU(Hidden1.Matmul(W2) + b2);
    YPred := Softmax(Hidden2.Matmul(W3) + b3, 1);

    loss := CrossEntropy(YPred, Y);
    loss.Backward();

    writeln('Loss pada iterasi ', i, '/', MAX_ITER, ': ', loss.Data.Item: 5: 2);

    optim.Step;
  end;

  WriteLn('Akurasi training:');
  PrintTensor(Mean(ArgMax(YPred, 1) = ArgMax(Y, 1)));

  DatasetTest := ReadCSV('../dataset/optdigits-test.csv');
  XTest := DatasetTest[[_ALL_, Range(0, 64)]] / 16;
  YTest := BinarizeLabel(DatasetTest[[_ALL_, 64]]);

  { Prediksi saat testing }
  Hidden1 := ReLU(XTest.Matmul(W1) + b1);  // dengan XTest
  Hidden2 := ReLU(Hidden1.Matmul(W2) + b2);
  YPred := Softmax(Hidden2.Matmul(W3) + b3, 1);
  WriteLn('Akurasi testing:');
  PrintTensor(Mean(ArgMax(YPred, 1) = ArgMax(YTest, 1)));

  ReadLn;
end.
