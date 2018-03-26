import java.util.*;

class Matrix
{

    double[][] matrix;

    public Matrix(double[][] matrix)
    {
        this.matrix = matrix;
    }

    public Matrix(ArrayList<Double> vector)
    {
        this.matrix = new double[vector.size()][1];
        for (int i = 0; i < vector.size(); i++)
        {
            this.matrix[i][0] = vector.get(i);
        }
    }

    /* Evaluate the inverse of a matrix */
    Matrix evaluateInverse()
    {
        int n = this.matrix.length;
        double d = this.determinant(n);

        if (d == 0)
        {
            System.out.println("\nInverse of Entered Matrix is not possible\n");
            System.exit(0);
        }

        Matrix cofactorMatrix = this.cofactor();

        Matrix inverseSigma = cofactorMatrix.transpose();

        for (int i = 0;i < n; i++)
        {
            for (int j = 0;j < n; j++)
            {
                inverseSigma.matrix[i][j] /= d;
            }
        }

        return inverseSigma;
    }

    /* Calculate the determinant */
    double determinant(int size)
    {
        double s = 1, det = 0;
        double[][] b = new double[size][size];
        int i, j, m, n, c;
        if (size == 1)
            return (this.matrix[0][0]);

        else
        {
            det = 0;
            for (c = 0; c < size; c++)
            {
                m = 0;
                n = 0;
                for (i = 0;i < size; i++)
                {
                    for (j = 0 ;j < size; j++)
                    {
                        b[i][j] = 0;
                        if (i != 0 && j != c)
                        {
                            b[m][n] = this.matrix[i][j];
                            if (n < (size - 2))
                                n++;
                            else
                            {
                                n = 0;
                                m++;
                            }
                        }
                    }
                }
                Matrix B = new Matrix(b);
                det = det + s * (this.matrix[0][c] * B.determinant(size-1));
                s = -1 * s;
            }
        }

        return (det);
    }

    /* Find the cofactor matrix */
    Matrix cofactor()
    {
        int f = this.matrix.length;
        double[][] b = new double[f][f];
        double[][] fac = new double[f][f];
        int p, q, m, n, i, j;
        for (q = 0;q < f; q++)
        {
            for (p = 0;p < f; p++)
            {
                m = 0;
                n = 0;
                for (i = 0;i < f; i++)
                {
                    for (j = 0;j < f; j++)
                    {
                        if (i != q && j != p)
                        {
                            b[m][n] = this.matrix[i][j];

                            if (n < (f - 2))
                                n++;
                            else
                            {
                                n = 0;
                                m++;
                            }
                        }
                    }
                }
                Matrix B = new Matrix(b);
                fac[q][p] = (double)Math.pow(-1, q + p) * B.determinant(b.length-1);
            }
        }

        return new Matrix(fac); // temporary fix TODO: fix transpose function
    }

    /* Find transpose of the matrix*/
    Matrix transpose()
    {
        int i, j;
        double[][] b = new double[this.matrix[0].length][this.matrix.length];

        for (i = 0;i < this.matrix.length; i++)
        {
            for (j = 0;j < this.matrix[0].length; j++)
            {
                b[j][i] = this.matrix[i][j];
            }
        }

        return new Matrix(b);
    }

    Matrix scalar_multiply(double scalar)
    {
        for (int i = 0; i < this.matrix.length; i++)
        {
            for (int j = 0; j < this.matrix[0].length; j++)
            {
                this.matrix[i][j] *= scalar;
            }
        }
        return this;
    }

    Matrix matrix_multiply(Matrix A)
    {
        int m = this.matrix.length;
        int p = this.matrix[0].length;
        int q = A.matrix[0].length;

        // System.out.println(m +" " +p +" " + q);

        double[][] result = new double[m][q];

        double sum = 0;
        for (int c = 0 ; c < m ; c++ )
         {
            for (int d = 0 ; d < q ; d++ )
            {
               for (int k = 0 ; k < p ; k++ )
               {
                  sum = sum + this.matrix[c][k]*A.matrix[k][d];
               }

               result[c][d] = sum;
               sum = 0;
            }
         }

         return new Matrix(result);
    }

    Matrix add(Matrix A)
    {
        int m = this.matrix.length;
        int n = this.matrix[0].length;

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                this.matrix[i][j] += A.matrix[i][j];
            }
        }
        return this;
    }

    void printMatrix()
    {
        System.out.println("----");
        for (int i = 0; i < this.matrix.length; i++)
        {
            for (int j = 0; j < this.matrix[0].length; j++)
            {
                System.out.print(this.matrix[i][j] + " ");
            }
            System.out.println();
        }
        System.out.println("----");
    }
}
