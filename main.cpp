#include<iostream>
#include<time.h>
#include<math.h>
#include<stdlib.h>
#include<fstream>

#define DBG 1

using namespace std;

double act(double z)
{
	return 1.0/(1.0+exp(-z));
}

double dact(double a)
{
	return a*(1.0-a);
}

double derror(double t,double a)
{
	return (a-t);
}

double mse(double *t,double *a,int n)
{
	double sum=0.0;
	for(int i=0;i<n;i++)
		sum+=0.5*pow(t[i]-a[i],2);
	return sum;
}

int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1=i&255;
	ch2=(i>>8)&255;
	ch3=(i>>16)&255;
	ch4=(i>>24)&255;
	return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}

int main(int argc,char* argv[])
{

	if(argc==1) cout<<argv[0]<<" n_layers [layers] n_epochs alpha";

	cout.precision(8);

	//internal variables
	int n_images_train;
	int n_images_test;
	double error_train,error_test,instance_error;

	//read binary files variable
	int magic_number;
	unsigned char temp;


	cout<<" Artificial Neural Network with n-layers";
	cout<<endl<<"========================================="<<endl<<endl;

	//read args
	int n_layers=strtol(argv[1],NULL,10);
	if(DBG) cout<<endl<<"n_layers = "<<n_layers;

	int *layer = new int[n_layers];
	for(int i=0;i<n_layers;i++)
	{
		layer[i]=strtol(argv[i+2],NULL,10);
		if(DBG) cout<<endl<<"layer["<<i<<"] = "<<layer[i];
	}

	int n_epoch=strtol(argv[n_layers+2],NULL,10);
	if(DBG) cout<<endl<<"n_epoch = "<<n_epoch;

	double alpha=strtod(argv[n_layers+3],NULL);
	if(DBG) cout<<endl<<"alpha = "<<alpha;

	//read MNIST input_train
	double **input_train;

	ifstream file_in_train("train-images.idx3-ubyte",ios::binary);
	if(file_in_train.is_open())
	{
		int n_rows=0;
		int n_cols=0;

		file_in_train.read((char*)&magic_number,sizeof(magic_number));
		magic_number=ReverseInt(magic_number);
		if(DBG) cout<<endl<<"magic_number in input_train = "<<magic_number;

		file_in_train.read((char*)&n_images_train,sizeof(n_images_train));
		n_images_train=ReverseInt(n_images_train);
		if(DBG) cout<<endl<<"n_images_train in input_train = "<<n_images_train;

		input_train = new double*[n_images_train];
		for(int i=0;i<n_images_train;i++)
			input_train[i] = new double[layer[0]];

		file_in_train.read((char*)&n_rows,sizeof(n_rows));
		n_rows= ReverseInt(n_rows);
		if(DBG) cout<<endl<<"n_rows in input_train = "<<n_rows;

		file_in_train.read((char*)&n_cols,sizeof(n_cols));
		n_cols= ReverseInt(n_cols);
		if(DBG) cout<<endl<<"n_cols in input_train = "<<n_cols;

		for(int i=0;i<n_images_train;i++)
			for(int r=0;r<n_rows;r++)
				for(int c=0;c<n_cols;c++)
				{
					file_in_train.read((char*)&temp,sizeof(temp));
					input_train[i][(n_rows*r)+c]=(double)temp/255.0;
				}
		file_in_train.close();
		printf("\nSuccess Reading input_train Data");
	}
	else
	{
		cout<<endl<<"Can not read input_train data ... !!!"<<endl;
		throw;
	}

	//read MNIST output_train
	double **output_train;
	ifstream file_out_train("train-labels.idx1-ubyte",ios::binary);
	if (file_out_train.is_open())
	{

		file_out_train.read((char*)&magic_number,sizeof(magic_number));
		magic_number=ReverseInt(magic_number);
		if(DBG) cout<<endl<<"magic_number in output_train = "<<magic_number;

		file_out_train.read((char*)&n_images_train,sizeof(n_images_train));
		n_images_train=ReverseInt(n_images_train);
		if(DBG) cout<<endl<<"n_images_train in output_train = "<<n_images_train;

		output_train = new double*[n_images_train];
		for(int i=0;i<n_images_train;i++)
		{
			output_train[i] = new double[layer[n_layers-1]];
			for(int j=0;j<layer[n_layers-1];j++)
				output_train[i][j]=0;
		}

		for(int i=0;i<n_images_train;i++)
		{
			file_out_train.read((char*)&temp,sizeof(temp));
			output_train[i][(int)temp]=1;
		}
		file_out_train.close();
		printf("\nSuccess Reading output_train Data");
	}
	else
	{
		cout<<endl<<"Can not read output_train data ... !!!"<<endl;
		throw;
	}
	//read MNIST input_test
	double **input_test;

	ifstream file_in_test("t10k-images.idx3-ubyte",ios::binary);
	if(file_in_test.is_open())
	{
		int n_rows=0;
		int n_cols=0;

		file_in_test.read((char*)&magic_number,sizeof(magic_number));
		magic_number=ReverseInt(magic_number);
		if(DBG) cout<<endl<<"magic_number in input_test = "<<magic_number;

		file_in_test.read((char*)&n_images_test,sizeof(n_images_test));
		n_images_test=ReverseInt(n_images_test);
		if(DBG) cout<<endl<<"n_images_test in input_test = "<<n_images_test;

		input_test = new double*[n_images_test];
		for(int i=0;i<n_images_test;i++)
			input_test[i] = new double[layer[0]];

		file_in_test.read((char*)&n_rows,sizeof(n_rows));
		n_rows= ReverseInt(n_rows);
		if(DBG) cout<<endl<<"n_rows in input_test = "<<n_rows;

		file_in_test.read((char*)&n_cols,sizeof(n_cols));
		n_cols= ReverseInt(n_cols);
		if(DBG) cout<<endl<<"n_cols in input_test = "<<n_cols;

		for(int i=0;i<n_images_test;i++)
			for(int r=0;r<n_rows;r++)
				for(int c=0;c<n_cols;c++)
				{
					file_in_test.read((char*)&temp,sizeof(temp));
					input_test[i][(n_rows*r)+c]=(double)temp/255.0;
				}
		file_in_test.close();
		printf("\nSuccess Reading input_test Data");
	}
	else
	{
		cout<<endl<<"Can not read input_test data ... !!!"<<endl;
		throw;
	}

	//read MNIST output_test
	double **output_test;
	ifstream file_out_test("t10k-labels.idx1-ubyte",ios::binary);
	if (file_out_test.is_open())
	{

		file_out_test.read((char*)&magic_number,sizeof(magic_number));
		magic_number=ReverseInt(magic_number);
		if(DBG) cout<<endl<<"magic_number in output_train = "<<magic_number;

		file_out_test.read((char*)&n_images_test,sizeof(n_images_test));
		n_images_test=ReverseInt(n_images_test);
		if(DBG) cout<<endl<<"n_images_test in output_train = "<<n_images_test;

		output_test = new double*[n_images_test];
		for(int i=0;i<n_images_test;i++)
		{
			output_test[i] = new double[layer[n_layers-1]];
			for(int j=0;j<layer[n_layers-1];j++)
				output_test[i][j]=0;
		}

		for(int i=0;i<n_images_test;i++)
		{
			file_out_test.read((char*)&temp,sizeof(temp));
			output_test[i][(int)temp]=1;
		}
		file_out_test.close();
		printf("\nSuccess Reading output_test Data");
	}
	else
	{
		cout<<endl<<"Can not read output_test data ... !!!"<<endl;
		throw;
	}

	//create structure of data for forward
	double z;
	double **a = new double*[n_layers];
	double **b = new double*[n_layers-1];
	double ***w = new double**[n_layers-1];
	for(int i=1;i<n_layers;i++)
		a[i] = new double[layer[i]];
	for(int i=0;i<n_layers-1;i++)
	{
		b[i] = new double[layer[i+1]];
		for(int j=0;j<layer[i+1];j++)
		{
			b[i][j]=((double)rand()/(RAND_MAX))/n_images_train;
			if(DBG) cout<<endl<<"b["<<i<<"]["<<j<<"] = "<<b[i][j];
		}
	}
	for(int i=0;i<n_layers-1;i++)
	{
		w[i] = new double*[layer[i]];
		for(int j=0;j<layer[i];j++)
		{
			w[i][j] = new double[layer[i+1]];
			for(int k=0;k<layer[i+1];k++)
			{
				w[i][j][k]=((double)rand()/(RAND_MAX))/n_images_train;
				//if(DBG) cout<<endl<<"w["<<i<<"]["<<j<<"]["<<k<<"] = "<<w[i][j][k];
			}
		}
	}

	//create structure of data for backward
	double **d_b = new double*[n_layers-1];
	for(int i=0;i<n_layers-1;i++)
		d_b[i] = new double[layer[i+1]];

	//execution in epochs
	for(int epoch=0;epoch<n_epoch;epoch++)
	{
		error_train=0.0;
		//for every instance in train
		for(int instance=0;instance<n_images_train;instance++)
		//for(int instance=0;instance<1;instance++)
		{
			//forward propagation with TRAIN DATA
			a[0]=input_train[instance];
			for(int i=0;i<n_layers-1;i++)
				for(int j=0;j<layer[i+1];j++)
				{
					z=b[i][j];
					for(int k=0;k<layer[i];k++)
						z+=a[i][k]*w[i][k][j];
					a[i+1][j]=act(z);
					if(DBG) cout<<endl<<"a["<<i+1<<"]["<<j<<"] = "<<a[i+1][j];
				}
			if(DBG){cout<<endl<<"target = "; for(int i=0;i<10;i++) cout<<output_train[instance][i];}

			//calculate error_train
			instance_error=mse(output_train[instance],a[n_layers-1],layer[n_layers-1]);
			if(DBG) cout<<endl<<"instance error_train "<<instance<<" = "<<instance_error;
			error_train+=instance_error;

			//backward propagation
			//for the last layer
			int m=n_layers-2;
			for(int i=0;i<layer[m+1];i++)
			{
				d_b[m][i]=derror(output_train[instance][i],a[m+1][i])*dact(a[m+1][i]);
				if(DBG) cout<<endl<<"d_b["<<m<<"]["<<i<<"] = "<<d_b[m][i];
			}

			//for another layers
			for(int m=n_layers-3;m>-1;m--)
				for(int i=0;i<layer[m+1];i++)
				{
					z=0.0;
					for(int j=0;j<layer[m+2];j++)
						z+=w[m+1][i][j]*d_b[m+1][j];
					d_b[m][i]=z*dact(a[m+1][i]);
					if(DBG) cout<<endl<<"d_b["<<m<<"]["<<i<<"] = "<<d_b[m][i];

				}

			//new weight and bias
			for(m=0;m<n_layers-1;m++)
				for(int j=0;j<layer[m+1];j++)
				{
					b[m][j]-=alpha*d_b[m][j];
					for(int i=0;i<layer[m];i++)
						w[m][i][j]-=alpha*d_b[m][j]*a[m][i];
				}
		}
		cout<<endl<<"Error en epoca para entrenamiento"<<epoch<<": "<<error_train;

		error_test=0.0;
		//for every instance in test
		for(int instance=0;instance<n_images_test;instance++)
		//for(int instance=0;instance<1;instance++)
		{
			//forward propagation with TEST DATA
			a[0]=input_test[instance];
			for(int i=0;i<n_layers-1;i++)
				for(int j=0;j<layer[i+1];j++)
				{
					z=b[i][j];
					for(int k=0;k<layer[i];k++)
						z+=a[i][k]*w[i][k][j];
					a[i+1][j]=act(z);
					if(DBG) cout<<endl<<"a["<<i+1<<"]["<<j<<"] = "<<a[i+1][j];
				}
			if(DBG){cout<<endl<<"target = "; for(int i=0;i<10;i++) cout<<output_test[instance][i];}

			//calculate error_test
			instance_error=mse(output_test[instance],a[n_layers-1],layer[n_layers-1]);
			if(DBG) cout<<endl<<"instance error_test"<<instance<<" = "<<instance_error;
			error_test+=instance_error;
		}

		cout<<endl<<"Error en epoca para test"<<epoch<<": "<<error_train;

	}

	return 0;
}
