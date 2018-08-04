#include<iostream>
#include<time.h>
#include<math.h>
#include<stdlib.h>
#include<fstream>

#define DBG 0

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
	cout.precision(8);

	//internal variables
	int n_images;
	double error,instance_error;
	//file read variables
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

/*	//read MNIST input
	double **input;
	ifstream file_in("train-images.idx3-ubyte",ios::binary);
	if(file_in.is_open())
	{
		int n_rows=0;
		int n_cols=0;

		file_in.read((char*)&magic_number,sizeof(magic_number));
		magic_number=ReverseInt(magic_number);
		if(DBG) cout<<endl<<"magic_number in input = "<<magic_number;

		file_in.read((char*)&n_images,sizeof(n_images));
		n_images=ReverseInt(n_images);
		if(DBG) cout<<endl<<"n_images in input = "<<n_images;

		input = new double*[n_images];
		for(int i=0;i<n_images;i++)
			input[i] = new double[layer[0]];

		file_in.read((char*)&n_rows,sizeof(n_rows));
		n_rows= ReverseInt(n_rows);
		if(DBG) cout<<endl<<"n_rows in input = "<<n_rows;

		file_in.read((char*)&n_cols,sizeof(n_cols));
		n_cols= ReverseInt(n_cols);
		if(DBG) cout<<endl<<"n_cols in input = "<<n_cols;

		for(int i=0;i<n_images;i++)
			for(int r=0;r<n_rows;r++)
				for(int c=0;c<n_cols;c++)
				{
					file_in.read((char*)&temp,sizeof(temp));
					input[i][(n_rows*r)+c]=(double)temp/255.0;
				}
		file_in.close();
		cout<<endl<<"Success Reading Input Data";
	}
	else
	{
		cout<<endl<<"Can not read input data ... !!!"<<endl;
		throw;
	}

	//read MNIST output
	double **output;
	ifstream file_out("train-labels.idx1-ubyte",ios::binary);
	if (file_out.is_open())
	{

		file_out.read((char*)&magic_number,sizeof(magic_number));
		magic_number=ReverseInt(magic_number);
		if(DBG) cout<<endl<<"magic_number in input = "<<magic_number;

		file_out.read((char*)&n_images,sizeof(n_images));
		n_images=ReverseInt(n_images);
		if(DBG) cout<<endl<<"n_images in input = "<<n_images;

		output = new double*[n_images];
		for(int i=0;i<n_images;i++)
		{
			output[i] = new double[layer[n_layers-1]];
			for(int j=0;j<layer[n_layers-1];j++)
				output[i][j]=0;
		}

		for(int i=0;i<n_images;i++)
		{
			file_out.read((char*)&temp,sizeof(temp));
			output[i][(int)temp]=1;
		}
		file_out.close();
		cout<<endl<<"Success Reading Output Data";
	}
	else
	{
		cout<<endl<<"Can not read output data ... !!!"<<endl;
		throw;
	}
*/

	n_images=1;
	double **input;
	input=new double*[n_images];
	input[0]=new double[2];
	double **output;
	output=new double*[n_images];
	output[0]=new double[2];

	input[0][0]=0.05;
	input[0][1]=0.1;
	output[0][0]=0.01;
	output[0][1]=0.99;


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
			b[i][j]=((double)rand()/(RAND_MAX));
	}
	for(int i=0;i<n_layers-1;i++)
	{
		w[i] = new double*[layer[i]];
		for(int j=0;j<layer[i];j++)
		{
			w[i][j] = new double[layer[i+1]];
			for(int k=0;k<layer[i+1];k++)
				w[i][j][k]=((double)rand()/(RAND_MAX));
		}
	}

	b[0][0]=0.35;
	b[0][1]=0.35;
	b[1][0]=0.60;
	b[1][1]=0.60;
	w[0][0][0]=0.15;
	w[0][0][1]=0.25;
	w[0][1][0]=0.20;
	w[0][1][1]=0.30;
	w[1][0][0]=0.40;
	w[1][0][1]=0.50;
	w[1][1][0]=0.45;
	w[1][1][1]=0.55;

	//create structure of data for backward
	double **d_b = new double*[n_layers-1];
	for(int i=0;i<n_layers-1;i++)
		d_b[i] = new double[layer[i+1]];

	//execution in epochs
	for(int epoch=0;epoch<n_epoch;epoch++)
	{
		error=0.0;
		//for every instance
		for(int instance=0;instance<n_images;instance++)
		{
			//forward propagation
			a[0]=input[instance];
			for(int i=0;i<n_layers-1;i++)
				for(int j=0;j<layer[i+1];j++)
				{
					z=b[i][j];
					for(int k=0;k<layer[i];k++)
						z+=a[i][k]*w[i][k][j];
					a[i+1][j]=act(z);
					if(DBG) cout<<endl<<"a["<<i+1<<"]["<<j<<"] = "<<a[i+1][j];
				}

			//calculate error
			instance_error=mse(output[instance],a[n_layers-1],layer[n_layers-1]);
			if(DBG) cout<<endl<<"instance error "<<instance<<" = "<<instance_error;
			error+=instance_error;

			//backward propagation
			//for the last layer
			int m=n_layers-2;
			for(int i=0;i<layer[m+1];i++)
			{
				d_b[m][i]=derror(output[instance][i],a[m+1][i])*dact(a[m+1][i]);
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
					//b[m][j]-=alpha*d_b[m][j];
					if(DBG) cout<<endl<<"b["<<m<<"]["<<j<<"] = "<<b[m][j];

					for(int i=0;i<layer[m];i++)
					{
						w[m][i][j]-=alpha*d_b[m][j]*a[m][i];
						if(DBG) cout<<endl<<"d_w["<<m<<"]["<<i<<"]["<<j<<"] = "<<alpha*d_b[m][j]*a[m][i];
						if(DBG) cout<<endl<<"w["<<m<<"]["<<i<<"]["<<j<<"] = "<<w[m][i][j];
					}
				}
		}
		cout<<endl<<"Error en epoca "<<epoch<<": "<<error;
	}

	return 0;
}

