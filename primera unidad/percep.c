#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define epoca 3000
#define K 0.3f

//0.00000
//Funcion de Entrenamiento Perceptron
float EntNt(float, float, float  );
//Funcion para las salidas 
float InitNt(float, float);
//Sigmoide
float sigmoide(float);
//pesos aleatorios
void pesos_initNt();

float Pesos[2];	
float Pesos2[2];
float Pesos3[2];
float bias=0.5f;
float bias2=0.5f;
float bias3=0.5f;
float Error;
 //                 1          1          1
 //                 0          1         0           
 //                 1          0         0
 //                0          0          0
float EntNt( float x0, float x1, float target )
{
  
//printf("x0=%f, x1=%f, t %f \n" ,x0, x1,  target );
  
  float net = 0;
  float out = 0;
  float delta[2];  
  float net2=0;
  float out2=0;
  float delta2[2];
  float net3=0;
  float out3=0;
  float delta3[2];
  //Es la variacion de los pesos sinapticos
  //float Error;
  
  float errorSalida;
  float errorN1;
  float errorN2;
   
  net = Pesos[0]*x0 + Pesos[1]*x1 - bias;
  net = sigmoide( net );
  out = net;

  net2 = Pesos2[0]*x0 + Pesos2[1]*x1 - bias2;
  net2 = sigmoide( net2 );
  out2 = net2;

  net3 = Pesos3[0]*out + Pesos3[1]*out2 - bias3;
  net3 = sigmoide( net3 );
  out3 = net3;
   
  Error = target - out3;
  //printf("Error funcion %f \n", Error); 

  errorSalida = Error * out3 * (1 - out3);
  errorN1 = errorSalida * Pesos3[0] * out * (1 - out);
  errorN2 = errorSalida * Pesos3[1] * out2 * (1 - out2);

  delta3[0] = K * errorSalida * out;
  delta3[1] = K * errorSalida * out2;
  Pesos3[0] += delta3[0];
  Pesos3[1] += delta3[1];
  bias3 -= K * errorSalida;

  bias -= K * errorN1;  //Como el bias es siempre 1, pongo que 
                    //el bias incluye ya su peso sinaptico
   
  delta[0] = K * errorN1 * x0;  //la variacion de los pesos sinapticos corresponde 
  delta[1] = K * errorN1 * x1;  //al error cometido, por la entrada correspondiente
    
   
  Pesos[0] += delta[0];  //Se ajustan los nuevos valores
  Pesos[1] += delta[1];  //de los pesos sinapticos

  delta2[0] = K * errorN2 * x0;
  delta2[1] = K * errorN2 * x1;
  Pesos2[0] += delta2[0];
  Pesos2[1] += delta2[1];
  bias2 -= K * errorN2;

   
  return out3;
}
 

 
float InitNt( float x0, float x1 )
{
  float net = 0;
  float out = 0;
  float net2 = 0;
  float out2 = 0;
  float net3 = 0;
  float out3 = 0;

//Pesos de cada epoca
//Peso 1 = 30.753101
//Peso 2 = 30.780966
//BiasBias = 61.583714
//Resultados 
//  net = 1.23*x0 + 2.4*x1+23;
//Peso 1 = 989.755493
//Peso 2 = -1407.284180
//Bias = 989.755981 

  net = Pesos[0]*x0 + Pesos[1]*x1 - bias;
  net = sigmoide( net );
  out = net;

  net2 = Pesos2[0]*x0 + Pesos2[1]*x1 - bias2;
  net2 = sigmoide( net2 );
  out2 = net2;

  net3 = Pesos3[0]*out + Pesos3[1]*out2 - bias3;
  net3 = sigmoide( net3 );
  out3 = net3;

  return out3;
}

 
 
void pesos_initNt(void)
{
int i;
  for(  i = 0; i < 2; i++ )
  {
    Pesos[i] = (float)rand()/RAND_MAX;
  }
  for(  i = 0; i < 2; i++ )
  {
    Pesos2[i] = (float)rand()/RAND_MAX;
  }
  for(  i = 0; i < 2; i++ )
  {
    Pesos3[i] = (float)rand()/RAND_MAX;
  }
}
 
float sigmoide( float s ){
  return (1/(1+ (-1*s)));
}

int main(){
  int i=0;
  float apr;
  pesos_initNt();
  
 while(i<epoca){
    
    printf("------------------------\n");
    printf("Salida Entrenamiento Epoca %d \n", i);
    apr=EntNt(1,1,0);
    printf("1,1=%f\n",apr);
    apr=EntNt(1,0,1);
    printf("1,0=%f\n",apr);
    apr=EntNt(0,1,1);
    printf("0,1=%f\n",apr);
    apr=EntNt(0,0,0);
    printf("0,0=%f\n",apr);
    printf("\n"); 
    printf("Pesos de cada epoca\n");
    printf("Peso 0 = %f\n", Pesos[0]);
    printf("Peso 1 = %f\n", Pesos[1]);
  
    printf("Bias = %f \n",bias);
	printf("Error %f\n ",Error  );
	printf("------------------------\n"); 
	i++;   
/*

    printf("Resultados\n");
    apr=InitNt(1,1);
    printf("1,1=%f\n",apr);
    apr=InitNt(1,0);
    printf("1,0=%f\n",apr);
    apr=InitNt(0,1);
    printf("0,1=%f\n",apr);
    apr=InitNt(0,0);
    printf("0,0=%f\n",apr);
*/

}

    printf("Resultados\n");
    apr=InitNt(1,1);
    printf("1,1=%f\n",apr);
    apr=InitNt(1,0);
    printf("1,0=%f\n",apr);
    apr=InitNt(0,1);
    printf("0,1=%f\n",apr);
    apr=InitNt(0,0);
    printf("0,0=%f\n",apr);

  return 0;
}
