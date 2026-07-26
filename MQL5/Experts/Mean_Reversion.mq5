//+------------------------------------------------------------------+
//|                                        Main Reversion - OakQuant |
//|                                         Copyright 2026, OakQuant |
//|                                             https://oakquant.com |
//+------------------------------------------------------------------+
#include <Mean_Reversion/EURGBP_H1_ONNX_include_0.mqh>
#include <Trade\Trade.mqh>
#include <Trade\AccountInfo.mqh>
#property strict
#property copyright "Copyright 2026, OakQuant"
#property link      "https://oakquant.com/"
#property version   "1.0"

CTrade mytrade;
CPositionInfo myposition;

input bool Allow_Buy = true;           //Allow BUY
input bool Allow_Sell = true;          //Allow SELL
double main_threshold = 0.5;
double meta_threshold = 0.5;
sinput double   MaximumRisk=0.001;     //Progressive lot coefficient
sinput double   ManualLot=0.01;        //Fixed lot, set 0 if progressive
sinput ulong    OrderMagic = 57633493; //Orders magic
input int max_orders = 3;              //Max positions number
input int orders_time_delay = 5;       //Time delay between positions
input int max_spread = 20;             //Max spread
input int stoploss = 2000;             //Stop loss
input int takeprofit = 200;            //Take profit
input string order_comment = "mean reversion bot";

static datetime last_time = 0;
#define Ask SymbolInfoDouble(_Symbol, SYMBOL_ASK)
#define Bid SymbolInfoDouble(_Symbol, SYMBOL_BID)


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
const long  ExtInputShape [] = {1, ArraySize(PeriodsEURGBP_H1_0)};
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
const long  ExtInputShape2 [] = {1, ArraySize(Periods_mEURGBP_H1_0)};
long     ExtHandle = INVALID_HANDLE, ExtHandle2 = INVALID_HANDLE;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   mytrade.SetExpertMagicNumber(OrderMagic);

// CPU-only: CatBoost exports ai.onnx.ml ops (TreeEnsembleClassifier/ZipMap)
// which only the CPU provider implements - GPU brings no gain and the CUDA
// provider bundled with MT5 (CUDA 13.x) fails on pre-Turing GPUs.
   ExtHandle = OnnxCreateFromBuffer(ExtModel_EURGBP_H1_0, ONNX_USE_CPU_ONLY);
   ExtHandle2 = OnnxCreateFromBuffer(ExtModel2_EURGBP_H1_0, ONNX_USE_CPU_ONLY);

   if(ExtHandle == INVALID_HANDLE || ExtHandle2 == INVALID_HANDLE)
     {
      Print("OnnxCreateFromBuffer error ", GetLastError());
      return(INIT_FAILED);
     }

   if(!OnnxSetInputShape(ExtHandle, 0, ExtInputShape))
     {
      Print("OnnxSetInputShape 1 failed, error ", GetLastError());
      OnnxRelease(ExtHandle);
      return(-1);
     }

   if(!OnnxSetInputShape(ExtHandle2, 0, ExtInputShape2))
     {
      Print("OnnxSetInputShape 2 failed, error ", GetLastError());
      OnnxRelease(ExtHandle2);
      return(-1);
     }

   const long output_shape[] = {1};
   if(!OnnxSetOutputShape(ExtHandle, 0, output_shape))
     {
      Print("OnnxSetOutputShape 1 error ", GetLastError());
      return(INIT_FAILED);
     }
   if(!OnnxSetOutputShape(ExtHandle2, 0, output_shape))
     {
      Print("OnnxSetOutputShape 2 error ", GetLastError());
      return(INIT_FAILED);
     }

   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   OnnxRelease(ExtHandle);
   OnnxRelease(ExtHandle2);
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   if(!isNewBar())
      return;

   double features[], features_m[];
   fill_araysEURGBP_H1_0(features);
   fill_arays_mEURGBP_H1_0(features_m);

   double f[ArraySize(PeriodsEURGBP_H1_0)], f_m[ArraySize(Periods_mEURGBP_H1_0)];

   for(int i = 0; i < ArraySize(PeriodsEURGBP_H1_0); i++)
     {
      f[i] = features[i];
     }

   for(int i = 0; i < ArraySize(Periods_mEURGBP_H1_0); i++)
     {
      f_m[i] = features_m[i];
     }

   static vector out(1), out_meta(1);

   struct output
     {
      long           label[];
      float          proba[];
     };

   output out2[], out2_meta[];

   OnnxRun(ExtHandle, ONNX_LOGLEVEL_INFO, f, out, out2);
   OnnxRun(ExtHandle2, ONNX_LOGLEVEL_INFO, f_m, out_meta, out2_meta);

   double sig = out2[0].proba[1];
   double meta_sig = out2_meta[0].proba[1];

// CLOSE ORDERS BY SIGNALS
   if((Ask-Bid < max_spread*_Point) && meta_sig > meta_threshold)
      if(countOrders(OrderMagic) > 0)
         for(int b = PositionsTotal() - 1; b >= 0; b--)
            if(PositionGetSymbol(b)==_Symbol)
              {
               // CLOSE BUY POSITIONS
               if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY &&
                  PositionGetInteger(POSITION_MAGIC) == OrderMagic &&
                  sig > main_threshold)
                  if(SymbolInfoInteger(_Symbol, SYMBOL_TRADE_FREEZE_LEVEL) <
                     MathAbs(Bid - PositionGetDouble(POSITION_PRICE_OPEN)))
                    {
                     int res = -1;
                     do
                       {
                        res = mytrade.PositionClose(_Symbol);
                        Sleep(50);
                       }
                     while(res == -1);
                    }
               // CLOSE SELL POSITIONS
               if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL &&
                  PositionGetInteger(POSITION_MAGIC) == OrderMagic &&
                  sig < 1-main_threshold)
                  if(SymbolInfoInteger(_Symbol, SYMBOL_TRADE_FREEZE_LEVEL) <
                     MathAbs(Bid - PositionGetDouble(POSITION_PRICE_OPEN)))
                    {
                     int res = -1;
                     do
                       {
                        res = mytrade.PositionClose(_Symbol);
                        Sleep(50);
                       }
                     while(res == -1);
                    }
              }

// OPEN POSITIONS BY SIGNALS
   if((Ask-Bid < max_spread*_Point) && meta_sig > meta_threshold &&
      AllowTrade(OrderMagic))
      if(countOrders(OrderMagic) < max_orders &&
         CheckMoneyForTrade(_Symbol, LotsOptimized(), ORDER_TYPE_BUY))
        {
         double l = LotsOptimized();
         if(sig < 1-main_threshold && Allow_Buy)
           {
            int res = -1;
            do
              {
               double stop = Bid - stoploss * _Point;
               double take = Ask + takeprofit * _Point;
               res = mytrade.PositionOpen(_Symbol, ORDER_TYPE_BUY, l, Ask, stop, take, order_comment);
               Sleep(50);
              }
            while(res == -1);
           }
         else
           {
            if(sig > main_threshold && Allow_Sell)
              {
               int res = -1;
               do
                 {
                  double stop = Ask + stoploss * _Point;
                  double take = Bid - takeprofit * _Point;
                  res = mytrade.PositionOpen(_Symbol, ORDER_TYPE_SELL, l, Bid, stop, take, order_comment);
                  Sleep(50);
                 }
               while(res == -1);
              }
           }
        }

//---

  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int countOrders(ulong magic) // Presence of the current positions on the market
  {
   int count=0;
   for(int i= PositionsTotal()-1; i>=0; i--)
     {
      if(PositionGetSymbol(i)==_Symbol)
        {
         if(PositionGetInteger(POSITION_MAGIC)==magic)
           {
            count++;
           }
        }
     }
   return(count);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool AllowTrade(ulong magic)
  {
   if(countOrders(OrderMagic)==0)
      return true;
   datetime last_pos = 0;
   if(countOrders(OrderMagic)!=0)
     {
      for(int b = PositionsTotal() - 1; b >= 0; b--)
         if(PositionGetSymbol(b)==_Symbol)
            if(PositionGetInteger(POSITION_MAGIC)==magic)
               if(PositionGetInteger(POSITION_TIME) > last_pos)
                  last_pos = (datetime)PositionGetInteger(POSITION_TIME);

      datetime time[];
      CopyTime(_Symbol,PERIOD_H1,0, 1, time);
      if(time[0] > last_pos + 3600 * orders_time_delay)
         return true;
     };

   return false;
  }
//+------------------------------------------------------------------+
double LotsOptimized()
  {
   double lot;

   if(MQLInfoInteger(MQL_OPTIMIZATION)==true)
     {
      lot=SymbolInfoDouble(Symbol(),SYMBOL_VOLUME_MIN);
      return lot;
     }
   CAccountInfo myaccount;
   SymbolInfoDouble(Symbol(),SYMBOL_VOLUME_STEP);

   lot=NormalizeDouble(myaccount.FreeMargin()*MaximumRisk/1000.0,2);
   if(ManualLot!=0.0)
      lot=ManualLot;

   double volume_step=SymbolInfoDouble(Symbol(),SYMBOL_VOLUME_STEP);
   int ratio=(int)MathRound(lot/volume_step);
   if(MathAbs(ratio*volume_step-lot)>0.0000001)
      lot=ratio*volume_step;

   if(lot<SymbolInfoDouble(Symbol(),SYMBOL_VOLUME_MIN))
      lot=SymbolInfoDouble(Symbol(),SYMBOL_VOLUME_MIN);
   if(lot>SymbolInfoDouble(Symbol(),SYMBOL_VOLUME_MAX))
      lot=SymbolInfoDouble(Symbol(),SYMBOL_VOLUME_MAX);
   return(lot);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool isNewBar()
  {
   datetime lastbar_time = datetime(SeriesInfoInteger(Symbol(), PERIOD_CURRENT, SERIES_LASTBAR_DATE));
   if(last_time == 0)
     {
      last_time = lastbar_time;
      return(false);
     }
   if(last_time != lastbar_time)
     {
      last_time = lastbar_time;
      return(true);
     }
   return(false);
  }
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CheckMoneyForTrade(string symb, double lots, ENUM_ORDER_TYPE type)
  {
   MqlTick mqltick;
   SymbolInfoTick(symb, mqltick);
   double price = mqltick.ask;
   if(type == ORDER_TYPE_SELL)
      price = mqltick.bid;
   double margin, free_margin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   if(!OrderCalcMargin(type, symb, lots, price, margin))
     {
      Print("Error in ", __FUNCTION__, " code=", GetLastError());
      return(false);
     }
   if(margin > free_margin)
     {
      Print("Not enough money for ", EnumToString(type), " ", lots, " ", symb, " Error code=", GetLastError());
      return(false);
     }
   return(true);
  }
//+------------------------------------------------------------------+
