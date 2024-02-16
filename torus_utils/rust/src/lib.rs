use numpy::ndarray::{ArrayBase, ArrayView1,  Dim, OwnedRepr};
use numpy::{Complex64, IntoPyArray, PyArray1};
use pyo3::{pymodule, types::PyModule, PyResult, Python};

// Hello this is pymodule
#[pymodule]
fn _rust_ext<'py>(_py: Python<'py>, m: &'py PyModule) -> PyResult<()> {
    fn refine(arr: ArrayView1<'_, Complex64>) -> ArrayBase<OwnedRepr<Complex64>, Dim<[usize; 1]>> {
        let phi = 2. * std::f64::consts::PI / 5.;
        let xi = Complex64::cis(phi);
        let mut arr_copy = arr.to_owned();

        let mut n = 1;
        while n < arr.dim() {
            let a1 = arr_copy[n - 1];
            let a2 = arr[n];

            let d0 = (a1 - a2 * xi.powu(0)).norm();
            let d1 = (a1 - a2 * xi.powu(1)).norm();
            let d2 = (a1 - a2 * xi.powu(2)).norm();
            let d3 = (a1 - a2 * xi.powu(3)).norm();
            let d4 = (a1 - a2 * xi.powu(4)).norm();

            let d_arr = vec![d0, d1, d2, d3, d4];
            let d_min: Option<usize> = d_arr
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(index, _)| index);

            match d_min {
                Some(s) => arr_copy[n] = arr_copy[n] * xi.powu(s.try_into().unwrap()),
                None => {}
            }

            n = n + 1;
        }
        arr_copy
    }

    /// add(a, b, /)
    /// --
    ///
    /// This function adds two unsigned 64-bit integers
    #[pyfn(m)]
    #[pyo3(name = "refine")]
    fn refine_py<'py>(py: Python<'py>, x: &'py PyArray1<Complex64>) -> &'py PyArray1<Complex64> {
        let x = unsafe { x.as_array() };
        // let x = unsafe { x.as_array_mut() };
        let out = refine(x).into_pyarray(py);
        out
    }

    Ok(())
}
