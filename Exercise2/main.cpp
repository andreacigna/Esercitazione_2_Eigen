#include <iostream>
#include <iomanip>
#include "Eigen/Eigen"

using namespace std;
using namespace Eigen;

int main()
{
    MatrixXd A1 = MatrixXd::Zero(2,2);
    Vector4d v1(5.547001962252291e-01, -3.770900990025203e-02,
               8.320502943378437e-01, -9.992887623566787e-01);
    A1 = v1.reshaped<RowMajor>(2,2);

    MatrixXd A2 = MatrixXd::Zero(2,2);
    Vector4d v2(5.547001962252291e-01, -5.540607316466765e-01,
                8.320502943378437e-01, -8.324762492991313e-01);
    A2 = v2.reshaped<RowMajor>(2,2);

    MatrixXd A3 = MatrixXd::Zero(2,2);
    Vector4d v3(5.547001962252291e-01, -5.547001955851905e-01,
               8.320502943378437e-01, -8.320502947645361e-01);
    A3 = v3.reshaped<RowMajor>(2,2);

    Vector2d b1(-5.169911863249772e-01, 1.672384680188350e-01);
    Vector2d b2(-6.394645785530173e-04, 4.259549612877223e-04);
    Vector2d b3(-6.400391328043042e-10, 4.266924591433963e-10);

    Vector2d xsol = -Vector2d::Ones();

    Vector2d X1lu = A1.lu().solve(b1);
    Vector2d X2lu = A2.lu().solve(b2);
    Vector2d X3lu = A3.lu().solve(b3);

    Vector2d X1QR = A1.fullPivHouseholderQr().solve(b1);
    Vector2d X2QR = A2.fullPivHouseholderQr().solve(b2);
    Vector2d X3QR = A3.fullPivHouseholderQr().solve(b3);

    cout << X1lu.transpose() << "\n" << X2lu.transpose() << "\n" << X3lu.transpose() << "\n" <<
        X1QR.transpose() << "\n" << X2QR.transpose() << "\n" << X3QR.transpose() << endl; //controllo che le soluzioni siano [-1;-1]

    cout << "\n" << "L'errore relativo della soluzione calcolata con la decomposizione PALU è: "
         << "\n" << scientific << setprecision(10) << (xsol-X1lu).norm()/xsol.norm() << " nel caso 1"
         << "\n" << (xsol-X2lu).norm()/xsol.norm() << " nel caso 2"
         << "\n" << (xsol-X3lu).norm()/xsol.norm() << " nel caso 3" << endl;

    cout << "\n" << "L'errore relativo della soluzione calcolata con la decomposizione QR è: "
         << "\n" << (xsol-X1QR).norm()/xsol.norm() << " nel caso 1"
         << "\n" << (xsol-X2QR).norm()/xsol.norm() << " nel caso 2"
         << "\n" << (xsol-X3QR).norm()/xsol.norm() << " nel caso 3" << endl;

  return 0;
}
