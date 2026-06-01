#include <Math\Stat\Math.mqh>
#resource "catmodel EURGBP_H1 0.onnx" as uchar ExtModel_EURGBP_H1_0[]
#resource "catmodel_m EURGBP_H1 0.onnx" as uchar ExtModel2_EURGBP_H1_0[]

int PeriodsEURGBP_H1_0[10] = {5,35,65,95,125,155,185,215,245,275};
int Periods_mEURGBP_H1_0[1] = {10};

void fill_araysEURGBP_H1_0( double &features[]) {
   double pr[], ret[];
   ArrayResize(ret, 1);
   for(int i=ArraySize(PeriodsEURGBP_H1_0)-1; i>=0; i--) {
       CopyClose(NULL,PERIOD_H1,1,PeriodsEURGBP_H1_0[i],pr);
       ret[0] = MathMean(pr);
       ArrayInsert(features, ret, ArraySize(features), 0, WHOLE_ARRAY); }
   ArraySetAsSeries(features, true);
}

void fill_arays_mEURGBP_H1_0( double &features[]) {
   double pr[], ret[];
   ArrayResize(ret, 1);
   for(int i=ArraySize(Periods_mEURGBP_H1_0)-1; i>=0; i--) {
       CopyClose(NULL,PERIOD_H1,1,Periods_mEURGBP_H1_0[i],pr);
       ret[0] = MathSkewness(pr);
       ArrayInsert(features, ret, ArraySize(features), 0, WHOLE_ARRAY); }
   ArraySetAsSeries(features, true);
}

