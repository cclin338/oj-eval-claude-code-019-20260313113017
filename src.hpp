#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());

  // Key optimization: maintain running concatenation of keys and values
  Matrix* k_concat = nullptr;
  Matrix* v_concat = nullptr;

  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();

    // Incrementally build K and V concatenations
    if (i == 0) {
      k_concat = matrix_memory_allocator.Allocate("k_concat");
      v_concat = matrix_memory_allocator.Allocate("v_concat");
      gpu_sim.Copy(keys[i], k_concat, kInGpuHbm);
      gpu_sim.Copy(values[i], v_concat, kInGpuHbm);
    } else {
      Matrix* k_temp = matrix_memory_allocator.Allocate("k_temp");
      Matrix* v_temp = matrix_memory_allocator.Allocate("v_temp");
      gpu_sim.Concat(k_concat, keys[i], k_temp, 0, kInGpuHbm);
      gpu_sim.Concat(v_concat, values[i], v_temp, 0, kInGpuHbm);
      gpu_sim.ReleaseMatrix(k_concat);
      gpu_sim.ReleaseMatrix(v_concat);
      k_concat = k_temp;
      v_concat = v_temp;
    }

    // Move everything to SRAM once
    gpu_sim.MoveMatrixToSharedMem(current_query);
    gpu_sim.MoveMatrixToSharedMem(k_concat);
    gpu_sim.MoveMatrixToSharedMem(v_concat);

    // Transpose K in place
    gpu_sim.Transpose(k_concat, kInSharedMemory);

    // Compute Q @ K.T
    Matrix* qk = matrix_memory_allocator.Allocate("qk");
    gpu_sim.MatMul(current_query, k_concat, qk);

    // Apply exp
    Matrix* exp_qk = matrix_memory_allocator.Allocate("exp_qk");
    gpu_sim.MatExp(qk, exp_qk);

    gpu_sim.ReleaseMatrix(qk);

    // Build softmax matrix row by row
    Matrix* softmax_qk = nullptr;

    for (size_t r = 0; r <= i; ++r) {
      Matrix* row_exp = matrix_memory_allocator.Allocate("row_exp");
      gpu_sim.GetRow(exp_qk, r, row_exp, kInSharedMemory);

      Matrix* row_sum = matrix_memory_allocator.Allocate("row_sum");
      gpu_sim.Sum(row_exp, row_sum);

      Matrix* row_softmax = matrix_memory_allocator.Allocate("row_softmax");
      gpu_sim.MatDiv(row_exp, row_sum, row_softmax);

      if (r == 0) {
        softmax_qk = matrix_memory_allocator.Allocate("softmax_qk");
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

    gpu_sim.ReleaseMatrix(exp_qk);

    // Compute softmax @ V
    Matrix* result = matrix_memory_allocator.Allocate("result");
    gpu_sim.MatMul(softmax_qk, v_concat, result);

    // Move result to HBM
    gpu_sim.MoveMatrixToGpuHbm(result);

    // Move k_concat and v_concat back to HBM for next iteration (they are modified by Transpose)
    gpu_sim.Transpose(k_concat, kInSharedMemory); // Transpose back to original form
    gpu_sim.MoveMatrixToGpuHbm(k_concat);
    gpu_sim.MoveMatrixToGpuHbm(v_concat);

    // Release temporary matrices
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