#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    /*
     * Implement your calculation logic here.
     * You can use the GpuSimulator instance to perform matrix operations.
     * For example:
     * gpu_sim.MoveMatrixToGpuHbm(keys[i]);
     * When your need a new matrix, to avoid memory leak, you should use
     * Matrix* new_matrix =
     * matrix_memory_allocator.Allocate(YOUR_MATRIX_NAME(string, which is
     * helpful for debugging)); It can manage the memory of matrices
     * automatically.
     */

    // Round i (0-indexed in C++):
    // - Q: shape [i+1, d]
    // - Need to concatenate K[0]...K[i] to get K_full: shape [i+1, d]
    // - Need to concatenate V[0]...V[i] to get V_full: shape [i+1, d]
    // - Compute: Softmax(Q @ K_full.T) @ V_full

    // Step 1: Concatenate all keys and values up to index i
    Matrix* k_concat = nullptr;
    Matrix* v_concat = nullptr;

    // Build K and V by concatenating in HBM (cheaper for concat)
    for (size_t j = 0; j <= i; ++j) {
      if (j == 0) {
        // Copy first key and value
        k_concat = matrix_memory_allocator.Allocate("k_concat");
        v_concat = matrix_memory_allocator.Allocate("v_concat");
        gpu_sim.Copy(keys[j], k_concat, kInGpuHbm);
        gpu_sim.Copy(values[j], v_concat, kInGpuHbm);
      } else {
        // Concatenate next key and value
        Matrix* k_temp = matrix_memory_allocator.Allocate("k_temp");
        Matrix* v_temp = matrix_memory_allocator.Allocate("v_temp");
        gpu_sim.Concat(k_concat, keys[j], k_temp, 0, kInGpuHbm);
        gpu_sim.Concat(v_concat, values[j], v_temp, 0, kInGpuHbm);
        gpu_sim.ReleaseMatrix(k_concat);
        gpu_sim.ReleaseMatrix(v_concat);
        k_concat = k_temp;
        v_concat = v_temp;
      }
    }

    // Step 2: Move Q, K, V to SRAM for computation
    gpu_sim.MoveMatrixToSharedMem(current_query);
    gpu_sim.MoveMatrixToSharedMem(k_concat);
    gpu_sim.MoveMatrixToSharedMem(v_concat);

    // Step 3: Transpose K in SRAM: K.T has shape [d, i+1]
    gpu_sim.Transpose(k_concat, kInSharedMemory);

    // Step 4: Q @ K.T: [i+1, d] @ [d, i+1] = [i+1, i+1]
    Matrix* qk = matrix_memory_allocator.Allocate("qk");
    gpu_sim.MatMul(current_query, k_concat, qk);

    // Step 5: Apply Softmax row-wise
    // For each row, compute exp then divide by sum
    Matrix* exp_qk = matrix_memory_allocator.Allocate("exp_qk");
    gpu_sim.MatExp(qk, exp_qk);

    // Softmax row-wise: for each row, divide by sum of that row
    // exp_qk has shape [i+1, i+1]
    // For each row r: softmax[r,c] = exp_qk[r,c] / sum(exp_qk[r,:])

    Matrix* softmax_qk = matrix_memory_allocator.Allocate("softmax_qk");

    // Process each row
    for (size_t r = 0; r <= i; ++r) {
      // Get row r from exp_qk
      Matrix* row_exp = matrix_memory_allocator.Allocate("row_exp");
      gpu_sim.GetRow(exp_qk, r, row_exp, kInSharedMemory);

      // Sum the row
      Matrix* row_sum = matrix_memory_allocator.Allocate("row_sum");
      gpu_sim.Sum(row_exp, row_sum);

      // Divide row by sum
      Matrix* row_softmax = matrix_memory_allocator.Allocate("row_softmax");
      gpu_sim.MatDiv(row_exp, row_sum, row_softmax);

      // Place the row back (we need to build softmax_qk row by row)
      if (r == 0) {
        gpu_sim.Copy(row_softmax, softmax_qk, kInSharedMemory);
      } else {
        Matrix* temp = matrix_memory_allocator.Allocate("temp");
        gpu_sim.Concat(softmax_qk, row_softmax, temp, 0, kInSharedMemory);
        gpu_sim.ReleaseMatrix(softmax_qk);
        softmax_qk = temp;
      }

      gpu_sim.ReleaseMatrix(row_exp);
      gpu_sim.ReleaseMatrix(row_sum);
      gpu_sim.ReleaseMatrix(row_softmax);
    }

    // Step 6: Softmax @ V: [i+1, i+1] @ [i+1, d] = [i+1, d]
    Matrix* result = matrix_memory_allocator.Allocate("result");
    gpu_sim.MatMul(softmax_qk, v_concat, result);

    // Move result to HBM
    gpu_sim.MoveMatrixToGpuHbm(result);

    // Release intermediate matrices
    gpu_sim.ReleaseMatrix(k_concat);
    gpu_sim.ReleaseMatrix(v_concat);
    gpu_sim.ReleaseMatrix(qk);
    gpu_sim.ReleaseMatrix(exp_qk);
    gpu_sim.ReleaseMatrix(softmax_qk);

    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*result);
    /*********************  End of your code *********************/
  
    /*
     * If you want to print debug information, you can use:
     * gpu_sim.Run(true, &matrix_memory_allocator);
     * At the end of your calculation, you should commit the answer:
     * rater.CommitAnswer(YOUR_ANSWER_MATRIX) in each iteration.
     * Your answer matrix should be in GPU HBM.
     * After the answer is committed, the answer matrix will be released
     * automatically.
     */
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu