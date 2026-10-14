OPENQASM 2.0;
include "qelib1.inc";
// Standard QASM 2 does not define rzz: declare its CX/RZ/CX realization.
gate rzz(theta) a,b {
  cx a,b;
  rz(theta) b;
  cx a,b;
}
qreg q[2];
h q[0];
h q[1];
rzz(0.3) q[0],q[1];
