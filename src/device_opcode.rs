#[derive(Debug, PartialEq, Clone, Copy)]
pub enum DeviceOpCode {
    AddF,
    AddD,
    AddI,

    SubF,
    SubD,
    SubI,

    MulF,
    MulD,
    MulI,

    DivF,
    DivD,
    DivI,
    
    MatMulF,
    MatMulD,

    Conv2DF,
    Conv2DD,

    Transpose,
}
