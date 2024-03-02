function [del_sigma] = RK4V2_delS(del_eps, epsilon, ndir, E_m, nu_m, Ax, Bx, theta0, E_f, nu_f, v_f)
%This function computes the stress increment using a 1 step RK4 algorithm
%without loops.
if ndir == 1
    del_e11 = del_eps(1);
    del_e22 = del_eps(2);
    del_e12 = del_eps(3);

    e11 = epsilon(1);
    e22 = epsilon(2);
    e12 = epsilon(3);


    del_sigma = [del_e22*((E_m*nu_m*(v_f - 1))/((2*nu_m - 1)*(nu_m + 1)) + (4*e22*v_f*sin(theta0/2)^4*(E_f - E_m)*(e11 + 1)^3*(nu_f - 1))/(2*nu_f^2 + nu_f - 1)) + del_e11*(v_f*((Bx*((2*theta0*(cos(theta0)/2 - 1/2))/(asin(sin(theta0/2)*(e11 + 1))^2*((cos(theta0) - 1)*(e11 + 1)^2 + 2)) - (sin(theta0/2)^3*(2*e11 + 2))/(1 - sin(theta0/2)^2*(e11 + 1)^2)^(3/2) + (theta0*sin(theta0/2)^3*(2*e11 + 2))/(2*asin(sin(theta0/2)*(e11 + 1))*(1 - sin(theta0/2)^2*(e11 + 1)^2)^(3/2))))/theta0 - (8*Ax*(cos(theta0)/2 - 1/2))/((cos(theta0) - 1)*(e11 + 1)^2 + 2) - (Ax*sin(theta0/2)^3*(2*e11 + 2)*(theta0 - 2*asin(sin(theta0/2)*(e11 + 1))))/(1 - sin(theta0/2)^2*(e11 + 1)^2)^(3/2) + (6*e22^2*sin(theta0/2)^4*(E_f - E_m)*(e11 + 1)^2*(nu_f - 1))/(2*nu_f^2 + nu_f - 1) + (4*e22^2*sin(theta0/2)^6*(E_f - E_m)*(e11 + 1)^4*(nu_f - 1))/((2*nu_f^2 + nu_f - 1)*(4*e11*sin(theta0/2)^2 + 2*sin(theta0/2)^2 + 2*e11^2*sin(theta0/2)^2 - 2)) - (e22^2*sin(theta0/2)^6*(2*e11 + 2)*(E_f - E_m)*(e11 + 1)^3*(nu_f - 1))/((sin(theta0/2)^2*(e11 + 1)^2 - 1)*(2*nu_f^2 + nu_f - 1))) - (E_m*(nu_m - 1)*(v_f - 1))/(2*nu_m^2 + nu_m - 1));...
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  del_e11*((E_m*nu_m*(v_f - 1))/((2*nu_m - 1)*(nu_m + 1)) + (4*e22*v_f*sin(theta0/2)^4*(E_f - E_m)*(e11 + 1)^3*(nu_f - 1))/(2*nu_f^2 + nu_f - 1)) - del_e22/(((v_f - 1)*(2*nu_m^2 + nu_m - 1))/(E_m*(nu_m - 1)) - (v_f*(2*nu_f^2 + nu_f - 1))/(E_f*(E_m/E_f + (sin(theta0/2)^4*(E_f - E_m)*(e11 + 1)^4)/E_f)*(nu_f - 1)));...
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     -del_e12/(((2*nu_m + 2)*(v_f - 1))/(2*E_m) - (v_f*(2*nu_f + 2))/E_f)];
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
end


if ndir == 2
    del_e11 = del_eps(2);
    del_e22 = del_eps(1);
    del_e12 = del_eps(3);

    e11 = epsilon(2);
    e22 = epsilon(1);
    e12 = epsilon(3);


    del_sigma = [                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  del_e11*((E_m*nu_m*(v_f - 1))/((2*nu_m - 1)*(nu_m + 1)) + (4*e22*v_f*sin(theta0/2)^4*(E_f - E_m)*(e11 + 1)^3*(nu_f - 1))/(2*nu_f^2 + nu_f - 1)) - del_e22/(((v_f - 1)*(2*nu_m^2 + nu_m - 1))/(E_m*(nu_m - 1)) - (v_f*(2*nu_f^2 + nu_f - 1))/(E_f*(E_m/E_f + (sin(theta0/2)^4*(E_f - E_m)*(e11 + 1)^4)/E_f)*(nu_f - 1)));...
                    del_e22*((E_m*nu_m*(v_f - 1))/((2*nu_m - 1)*(nu_m + 1)) + (4*e22*v_f*sin(theta0/2)^4*(E_f - E_m)*(e11 + 1)^3*(nu_f - 1))/(2*nu_f^2 + nu_f - 1)) + del_e11*(v_f*((Bx*((2*theta0*(cos(theta0)/2 - 1/2))/(asin(sin(theta0/2)*(e11 + 1))^2*((cos(theta0) - 1)*(e11 + 1)^2 + 2)) - (sin(theta0/2)^3*(2*e11 + 2))/(1 - sin(theta0/2)^2*(e11 + 1)^2)^(3/2) + (theta0*sin(theta0/2)^3*(2*e11 + 2))/(2*asin(sin(theta0/2)*(e11 + 1))*(1 - sin(theta0/2)^2*(e11 + 1)^2)^(3/2))))/theta0 - (8*Ax*(cos(theta0)/2 - 1/2))/((cos(theta0) - 1)*(e11 + 1)^2 + 2) - (Ax*sin(theta0/2)^3*(2*e11 + 2)*(theta0 - 2*asin(sin(theta0/2)*(e11 + 1))))/(1 - sin(theta0/2)^2*(e11 + 1)^2)^(3/2) + (6*e22^2*sin(theta0/2)^4*(E_f - E_m)*(e11 + 1)^2*(nu_f - 1))/(2*nu_f^2 + nu_f - 1) + (4*e22^2*sin(theta0/2)^6*(E_f - E_m)*(e11 + 1)^4*(nu_f - 1))/((2*nu_f^2 + nu_f - 1)*(4*e11*sin(theta0/2)^2 + 2*sin(theta0/2)^2 + 2*e11^2*sin(theta0/2)^2 - 2)) - (e22^2*sin(theta0/2)^6*(2*e11 + 2)*(E_f - E_m)*(e11 + 1)^3*(nu_f - 1))/((sin(theta0/2)^2*(e11 + 1)^2 - 1)*(2*nu_f^2 + nu_f - 1))) - (E_m*(nu_m - 1)*(v_f - 1))/(2*nu_m^2 + nu_m - 1));...
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     -del_e12/(((2*nu_m + 2)*(v_f - 1))/(2*E_m) - (v_f*(2*nu_f + 2))/E_f)];
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
end
end

