program data_loading;

uses
  MultiArray, numerik;

var
  Dataset, A, B: TMultiArray;

begin
  A := TMultiArray([1, 2, 3, 4]).Reshape([2, 2]);
  B := 100;

  WriteLn('A:');
  PrintMultiArray(A);

  WriteLn('A transpose:');
  PrintMultiArray(A.T);

  WriteLn('B:');
  PrintMultiArray(B);

  WriteLn('A + B:');
  PrintMultiArray(A + B);

  WriteLn('Sin(A^2):');
  PrintMultiArray(Sin(A ** 2));

  ReadLn;

  Dataset := ReadCSV('../dataset/mall_customers.csv');
  PrintMultiArray(Dataset);
  ReadLn;
end.

