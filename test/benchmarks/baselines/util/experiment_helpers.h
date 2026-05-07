#include <stdio.h>
#include <cstring>
#include <random>
#include <vector>
#include <numeric>
#include <algorithm>
#include <thread>
#include <time.h>
#include <pthread.h>
#include <iostream>

#define TEST_FOR_EQUIVALENCE 0
#define CPU_THREADS_GLOBAL 8
#define TEST_ITERS_GLOBAL 2

#define RANDOM_MEAN 0
#define RANDOM_STDEV 1

#if TEST_FOR_EQUIVALENCE
   std::default_random_engine randEng(1337); //seed
#else
   std::default_random_engine randEng(time(0)); //seed
#endif
std::normal_distribution<double> randDist(RANDOM_MEAN, RANDOM_STDEV); //mean followed by stdiv
template <typename T>
T getRand(){return static_cast<T>(randDist(randEng));}

template<bool PRINT_DISTRIBUTION = true>
void printStats(std::vector<double> *times){
   double sum = std::accumulate(times->begin(), times->end(), 0.0);
   double mean = sum/static_cast<double>(times->size());
   std::vector<double> diff(times->size());
   std::transform(times->begin(), times->end(), diff.begin(), [mean](double x) {return x - mean;});
   double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
   double stdev = std::sqrt(sq_sum / times->size());
   std::vector<double>::iterator minInd = std::min_element(times->begin(), times->end());
   std::vector<double>::iterator maxInd = std::max_element(times->begin(), times->end());
   double min = times->at(std::distance(times->begin(), minInd)); 
   double max = times->at(std::distance(times->begin(), maxInd));
   printf("Average[%fus] Std Dev [%fus] Min [%fus] Max [%fus] \n",mean,stdev,min,max);
   if (PRINT_DISTRIBUTION){
      double hist[] = {0,0,0,0,0,0,0};
      for(int i = 0; i < times->size(); i++){
         double value = times->at(i);
         if (value < mean - stdev){
            if (value < mean - 2*stdev){
               if (value < mean - 3*stdev){hist[0] += 1.0;}
               else{hist[1] += 1.0;}
            }
            else{hist[2] += 1.0;}
         }
         else if (value > mean + stdev){
            if (value > mean + 2*stdev){
               if (value > mean + 3*stdev){hist[6] += 1.0;}
               else{hist[5] += 1.0;}
            }
            else{hist[4] += 1.0;}
         }
         else{hist[3] += 1.0;}
      }
      for(int i = 0; i < 7; i++){hist[i] = (hist[i]/static_cast<double>(times->size()))*100;}
      printf("    Distribution |  -3  |  -2  |  -1  |   0  |   1  |   2  |   3  |\n");
      printf("    (X std dev)  | %2.2f | %2.2f | %2.2f | %2.2f | %2.2f | %2.2f | %2.2f |\n",
                                hist[0],hist[1],hist[2],hist[3],hist[4],hist[5],hist[6]);
      std::sort(times->begin(), times->end()); 
      printf("    Percentiles |  50   |  60   |  70   |  75   |  80   |  85   |  90   |  95   |  99   |\n");
      printf("                | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f |\n",
                              times->at(times->size()/2),times->at(times->size()/5*3),times->at(times->size()/10*7),
                              times->at(times->size()/4*3),times->at(times->size()/5*4),times->at(times->size()/20*17),
                              times->at(times->size()/10*9),times->at(times->size()/20*19),times->at(times->size()/100*99));
      bool onePer = false; bool twoPer = false; bool fivePer = false; bool tenPer = false;
      for(int i = 0; i < times->size(); i++){
         if(!onePer && times->at(i) >= mean * 1.01){ onePer = true;
            printf("    More than 1 Percent above mean at [%2.2f] Percentile\n",static_cast<double>(i)/static_cast<double>(times->size())*100.0);
         }
         if(!twoPer && times->at(i) >= mean * 1.02){ twoPer = true;
            printf("    More than 2 Percent above mean at [%2.2f] Percentile\n",static_cast<double>(i)/static_cast<double>(times->size())*100.0);
         }
         if(!fivePer && times->at(i) >= mean * 1.05){ fivePer = true;
            printf("    More than 5 Percent above mean at [%2.2f] Percentile\n",static_cast<double>(i)/static_cast<double>(times->size())*100.0);
         }
         if(!tenPer && times->at(i) >= mean * 1.10){ tenPer = true;
            printf("    More than 10 Percent above mean at [%2.2f] Percentile\n",static_cast<double>(i)/static_cast<double>(times->size())*100.0);
         }
      }
   }
}
