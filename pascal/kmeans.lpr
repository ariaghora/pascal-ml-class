program kmeans;

{$mode objfpc}{$H+}

uses
  multiarray,
  unsupervised.kmeans,
  gnuplot;

var
  KMeansClust: TKMeans;
  Labels, DataSet: TMultiArray;
  fig: TFigure;

begin
  DataSet := ReadCSV('../dataset/mall_customers.csv');

  fig := TFigure.Create('Mall Customers', 'Pendapatan Per-tahun (ribu USD)',
    'Spending score');
  fig.AddScatterPlot(DataSet[[_ALL_, [1, 2]]], '');
  fig.Show;

  KMeansClust := TKMeans.Create(5);
  KMeansClust.Fit(DataSet);

  Labels := KMeansClust.Predict(DataSet);
  PrintMultiArray(Labels);

  readln;

end.
