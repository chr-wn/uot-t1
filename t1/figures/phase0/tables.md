## Accuracy by condition (candidate readout; bootstrap 95% CI over templates × items)

| model          | task   | version   | condition     |   n |   acc |   ci_lo |   ci_hi |   chance |   acc_gen_dim|parsed |   acc_inorder |   acc_swapped |   acc_length_matched |
|:---------------|:-------|:----------|:--------------|----:|------:|--------:|--------:|---------:|---------------------:|--------------:|--------------:|---------------------:|
| gemma2-9b      | T1     | v2        | FAM-NAMED     | 480 | 0.983 |   0.959 |   0.998 |    0.25  |                0.786 |         0.993 |         0.969 |                0.989 |
| gemma2-9b      | T1     | v2        | FAM-UNNAMED   | 480 | 0.865 |   0.777 |   0.93  |    0.25  |                0.796 |         0.872 |         0.855 |                0.875 |
| gemma2-9b      | T1     | v2        | INV-BASE      | 480 | 0.904 |   0.82  |   0.979 |    0.25  |                0.986 |         0.915 |         0.891 |                0.905 |
| gemma2-9b      | T1     | v2        | INV-BASE-TWIN | 480 | 0.902 |   0.823 |   0.973 |    0.25  |                0.934 |         0.946 |         0.851 |                0.924 |
| gemma2-9b      | T1     | v2        | INV-LEX       | 480 | 0.892 |   0.839 |   0.937 |    0.25  |                0.435 |         0.896 |         0.886 |                0.92  |
| gemma2-9b      | T1     | v2        | INV-LEX-TWIN  | 480 | 0.919 |   0.873 |   0.955 |    0.25  |                0.779 |         0.925 |         0.91  |                0.934 |
| gemma2-9b      | T2     | v1        | FAM-DIFF-FAR  | 240 | 0.521 |   0.432 |   0.619 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| gemma2-9b      | T2     | v1        | FAM-NEAR-MISS | 240 | 0.5   |   0.39  |   0.606 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| gemma2-9b      | T2     | v1        | FAM-SAME-DIM  | 240 | 0.521 |   0.423 |   0.623 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| gemma2-9b      | T2     | v1        | FAM-SAME-UNIT | 240 | 0.546 |   0.435 |   0.665 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| gemma2-9b      | T2     | v1        | INV-BASE      | 240 | 0.521 |   0.443 |   0.593 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| gemma2-9b      | T2     | v1        | INV-LEX       | 240 | 0.5   |   0.39  |   0.612 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| gemma2-9b      | T3     | v1        | FAM-NAMED     | 240 | 0.8   |   0.733 |   0.868 |    0.25  |              nan     |       nan     |       nan     |              nan     |
| gemma2-9b      | T3     | v1        | FAM-UNNAMED   | 240 | 0.525 |   0.419 |   0.625 |    0.25  |              nan     |       nan     |       nan     |              nan     |
| gemma2-9b      | T3     | v1        | INV-LEX       | 240 | 0.604 |   0.496 |   0.699 |    0.25  |              nan     |       nan     |       nan     |              nan     |
| gemma2-9b      | T4     | v1        | FAM-DIFF-FAR  | 240 | 0.517 |   0.448 |   0.585 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| gemma2-9b      | T4     | v1        | FAM-DIFF-NEAR | 240 | 0.529 |   0.418 |   0.639 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| gemma2-9b      | T4     | v1        | FAM-NEAR-MISS | 240 | 0.508 |   0.439 |   0.582 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| gemma2-9b      | T4     | v1        | FAM-SAME-DIM  | 240 | 0.517 |   0.441 |   0.589 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| gemma2-9b      | T4     | v1        | INV-LEX       | 240 | 0.5   |   0.426 |   0.571 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| gemma2-9b      | T5     | v1        | FAM-FAR       | 240 | 0.338 |   0.262 |   0.422 |    0.333 |              nan     |       nan     |       nan     |              nan     |
| gemma2-9b      | T5     | v1        | FAM-NEAR      | 240 | 0.342 |   0.281 |   0.407 |    0.333 |              nan     |       nan     |       nan     |              nan     |
| gemma2-9b      | T5     | v1        | INV-LEX       | 240 | 0.329 |   0.262 |   0.397 |    0.333 |              nan     |       nan     |       nan     |              nan     |
| olmo3-32b      | T1     | v2        | FAM-NAMED     | 480 | 0.983 |   0.967 |   0.996 |    0.25  |                0.797 |         0.986 |         0.979 |                0.983 |
| olmo3-32b      | T1     | v2        | FAM-UNNAMED   | 480 | 0.906 |   0.84  |   0.956 |    0.25  |                0.748 |         0.883 |         0.935 |                0.913 |
| olmo3-32b      | T1     | v2        | INV-BASE      | 480 | 0.912 |   0.848 |   0.973 |    0.25  |                0.953 |         0.907 |         0.919 |                0.918 |
| olmo3-32b      | T1     | v2        | INV-BASE-TWIN | 480 | 0.938 |   0.885 |   0.983 |    0.25  |                0.975 |         0.965 |         0.905 |                0.957 |
| olmo3-32b      | T1     | v2        | INV-LEX       | 480 | 0.952 |   0.914 |   0.983 |    0.25  |                0.444 |         0.961 |         0.94  |                0.967 |
| olmo3-32b      | T1     | v2        | INV-LEX-TWIN  | 480 | 0.942 |   0.904 |   0.975 |    0.25  |                0.76  |         0.943 |         0.94  |                0.954 |
| olmo3-32b      | T2     | v1        | FAM-DIFF-FAR  | 240 | 0.558 |   0.44  |   0.708 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| olmo3-32b      | T2     | v1        | FAM-NEAR-MISS | 240 | 0.567 |   0.451 |   0.69  |    0.5   |              nan     |       nan     |       nan     |              nan     |
| olmo3-32b      | T2     | v1        | FAM-SAME-DIM  | 240 | 0.496 |   0.389 |   0.627 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| olmo3-32b      | T2     | v1        | FAM-SAME-UNIT | 240 | 0.658 |   0.532 |   0.784 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| olmo3-32b      | T2     | v1        | INV-BASE      | 240 | 0.525 |   0.434 |   0.643 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| olmo3-32b      | T2     | v1        | INV-LEX       | 240 | 0.504 |   0.392 |   0.613 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| olmo3-32b      | T3     | v1        | FAM-NAMED     | 240 | 0.933 |   0.899 |   0.962 |    0.25  |              nan     |       nan     |       nan     |              nan     |
| olmo3-32b      | T3     | v1        | FAM-UNNAMED   | 240 | 0.783 |   0.727 |   0.833 |    0.25  |              nan     |       nan     |       nan     |              nan     |
| olmo3-32b      | T3     | v1        | INV-LEX       | 240 | 0.738 |   0.615 |   0.847 |    0.25  |              nan     |       nan     |       nan     |              nan     |
| olmo3-32b      | T4     | v1        | FAM-DIFF-FAR  | 240 | 0.888 |   0.824 |   0.944 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| olmo3-32b      | T4     | v1        | FAM-DIFF-NEAR | 240 | 0.771 |   0.703 |   0.831 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| olmo3-32b      | T4     | v1        | FAM-NEAR-MISS | 240 | 0.842 |   0.787 |   0.894 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| olmo3-32b      | T4     | v1        | FAM-SAME-DIM  | 240 | 0.846 |   0.796 |   0.893 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| olmo3-32b      | T4     | v1        | INV-LEX       | 240 | 0.671 |   0.536 |   0.784 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| olmo3-32b      | T5     | v1        | FAM-FAR       | 240 | 0.512 |   0.382 |   0.657 |    0.333 |              nan     |       nan     |       nan     |              nan     |
| olmo3-32b      | T5     | v1        | FAM-NEAR      | 240 | 0.454 |   0.37  |   0.533 |    0.333 |              nan     |       nan     |       nan     |              nan     |
| olmo3-32b      | T5     | v1        | INV-LEX       | 240 | 0.458 |   0.314 |   0.633 |    0.333 |              nan     |       nan     |       nan     |              nan     |
| olmo3-7b       | T1     | v2        | FAM-NAMED     | 480 | 0.969 |   0.943 |   0.989 |    0.25  |                0.788 |         0.976 |         0.959 |                0.972 |
| olmo3-7b       | T1     | v2        | FAM-UNNAMED   | 480 | 0.86  |   0.795 |   0.913 |    0.25  |                0.677 |         0.831 |         0.897 |                0.864 |
| olmo3-7b       | T1     | v2        | INV-BASE      | 480 | 0.879 |   0.768 |   0.976 |    0.25  |                0.72  |         0.876 |         0.882 |                0.891 |
| olmo3-7b       | T1     | v2        | INV-BASE-TWIN | 480 | 0.9   |   0.826 |   0.969 |    0.25  |                0.927 |         0.934 |         0.86  |                0.924 |
| olmo3-7b       | T1     | v2        | INV-LEX       | 480 | 0.91  |   0.865 |   0.95  |    0.25  |                0.361 |         0.921 |         0.896 |                0.923 |
| olmo3-7b       | T1     | v2        | INV-LEX-TWIN  | 480 | 0.923 |   0.877 |   0.962 |    0.25  |                0.747 |         0.932 |         0.91  |                0.947 |
| olmo3-7b       | T2     | v1        | FAM-DIFF-FAR  | 240 | 0.604 |   0.494 |   0.721 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| olmo3-7b       | T2     | v1        | FAM-NEAR-MISS | 240 | 0.55  |   0.436 |   0.656 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| olmo3-7b       | T2     | v1        | FAM-SAME-DIM  | 240 | 0.571 |   0.47  |   0.671 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| olmo3-7b       | T2     | v1        | FAM-SAME-UNIT | 240 | 0.733 |   0.613 |   0.864 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| olmo3-7b       | T2     | v1        | INV-BASE      | 240 | 0.5   |   0.427 |   0.576 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| olmo3-7b       | T2     | v1        | INV-LEX       | 240 | 0.579 |   0.442 |   0.688 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| olmo3-7b       | T3     | v1        | FAM-NAMED     | 240 | 0.767 |   0.693 |   0.838 |    0.25  |              nan     |       nan     |       nan     |              nan     |
| olmo3-7b       | T3     | v1        | FAM-UNNAMED   | 240 | 0.579 |   0.512 |   0.651 |    0.25  |              nan     |       nan     |       nan     |              nan     |
| olmo3-7b       | T3     | v1        | INV-LEX       | 240 | 0.596 |   0.529 |   0.663 |    0.25  |              nan     |       nan     |       nan     |              nan     |
| olmo3-7b       | T4     | v1        | FAM-DIFF-FAR  | 240 | 0.7   |   0.625 |   0.771 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| olmo3-7b       | T4     | v1        | FAM-DIFF-NEAR | 240 | 0.675 |   0.581 |   0.757 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| olmo3-7b       | T4     | v1        | FAM-NEAR-MISS | 240 | 0.692 |   0.601 |   0.798 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| olmo3-7b       | T4     | v1        | FAM-SAME-DIM  | 240 | 0.754 |   0.653 |   0.837 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| olmo3-7b       | T4     | v1        | INV-LEX       | 240 | 0.812 |   0.714 |   0.906 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| olmo3-7b       | T5     | v1        | FAM-FAR       | 240 | 0.304 |   0.229 |   0.376 |    0.333 |              nan     |       nan     |       nan     |              nan     |
| olmo3-7b       | T5     | v1        | FAM-NEAR      | 240 | 0.292 |   0.188 |   0.385 |    0.333 |              nan     |       nan     |       nan     |              nan     |
| olmo3-7b       | T5     | v1        | INV-LEX       | 240 | 0.3   |   0.224 |   0.379 |    0.333 |              nan     |       nan     |       nan     |              nan     |
| qwen3-14b-base | T1     | v2        | FAM-NAMED     | 480 | 0.981 |   0.961 |   0.996 |    0.25  |                0.749 |         0.986 |         0.974 |                0.978 |
| qwen3-14b-base | T1     | v2        | FAM-UNNAMED   | 480 | 0.881 |   0.81  |   0.942 |    0.25  |                0.791 |         0.876 |         0.888 |                0.875 |
| qwen3-14b-base | T1     | v2        | INV-BASE      | 480 | 0.935 |   0.872 |   0.989 |    0.25  |                0.923 |         0.958 |         0.91  |                0.934 |
| qwen3-14b-base | T1     | v2        | INV-BASE-TWIN | 480 | 0.94  |   0.89  |   0.984 |    0.25  |                0.82  |         0.958 |         0.919 |                0.959 |
| qwen3-14b-base | T1     | v2        | INV-LEX       | 480 | 0.965 |   0.934 |   0.989 |    0.25  |                0.659 |         0.978 |         0.945 |                0.97  |
| qwen3-14b-base | T1     | v2        | INV-LEX-TWIN  | 480 | 0.931 |   0.891 |   0.965 |    0.25  |                0.786 |         0.939 |         0.92  |                0.949 |
| qwen3-14b-base | T2     | v1        | FAM-DIFF-FAR  | 240 | 0.662 |   0.553 |   0.786 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-14b-base | T2     | v1        | FAM-NEAR-MISS | 240 | 0.658 |   0.515 |   0.787 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-14b-base | T2     | v1        | FAM-SAME-DIM  | 240 | 0.625 |   0.494 |   0.769 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-14b-base | T2     | v1        | FAM-SAME-UNIT | 240 | 0.875 |   0.776 |   0.957 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-14b-base | T2     | v1        | INV-BASE      | 240 | 0.671 |   0.509 |   0.833 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-14b-base | T2     | v1        | INV-LEX       | 240 | 0.629 |   0.453 |   0.784 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-14b-base | T3     | v1        | FAM-NAMED     | 240 | 0.992 |   0.976 |   1     |    0.25  |              nan     |       nan     |       nan     |              nan     |
| qwen3-14b-base | T3     | v1        | FAM-UNNAMED   | 240 | 0.904 |   0.859 |   0.945 |    0.25  |              nan     |       nan     |       nan     |              nan     |
| qwen3-14b-base | T3     | v1        | INV-LEX       | 240 | 0.958 |   0.917 |   0.992 |    0.25  |              nan     |       nan     |       nan     |              nan     |
| qwen3-14b-base | T4     | v1        | FAM-DIFF-FAR  | 240 | 0.875 |   0.793 |   0.949 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-14b-base | T4     | v1        | FAM-DIFF-NEAR | 240 | 0.842 |   0.775 |   0.9   |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-14b-base | T4     | v1        | FAM-NEAR-MISS | 240 | 0.708 |   0.64  |   0.771 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-14b-base | T4     | v1        | FAM-SAME-DIM  | 240 | 0.896 |   0.795 |   0.966 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-14b-base | T4     | v1        | INV-LEX       | 240 | 0.829 |   0.663 |   1     |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-14b-base | T5     | v1        | FAM-FAR       | 240 | 0.571 |   0.491 |   0.664 |    0.333 |              nan     |       nan     |       nan     |              nan     |
| qwen3-14b-base | T5     | v1        | FAM-NEAR      | 240 | 0.592 |   0.516 |   0.661 |    0.333 |              nan     |       nan     |       nan     |              nan     |
| qwen3-14b-base | T5     | v1        | INV-LEX       | 240 | 0.633 |   0.494 |   0.785 |    0.333 |              nan     |       nan     |       nan     |              nan     |
| qwen3-4b       | T1     | v2        | FAM-NAMED     | 480 | 0.84  |   0.781 |   0.896 |    0.25  |                0.854 |         0.846 |         0.83  |                0.847 |
| qwen3-4b       | T1     | v2        | FAM-UNNAMED   | 480 | 0.635 |   0.502 |   0.758 |    0.25  |                0.599 |         0.677 |         0.584 |                0.626 |
| qwen3-4b       | T1     | v2        | INV-BASE      | 480 | 0.892 |   0.806 |   0.963 |    0.25  |                0.935 |         0.931 |         0.846 |                0.885 |
| qwen3-4b       | T1     | v2        | INV-BASE-TWIN | 480 | 0.812 |   0.684 |   0.927 |    0.25  |                0.75  |         0.857 |         0.76  |                0.846 |
| qwen3-4b       | T1     | v2        | INV-LEX       | 480 | 0.844 |   0.789 |   0.896 |    0.25  |                0.634 |         0.867 |         0.811 |                0.86  |
| qwen3-4b       | T1     | v2        | INV-LEX-TWIN  | 480 | 0.769 |   0.7   |   0.831 |    0.25  |                0.743 |         0.799 |         0.726 |                0.78  |
| qwen3-4b       | T2     | v1        | FAM-DIFF-FAR  | 240 | 0.671 |   0.552 |   0.796 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-4b       | T2     | v1        | FAM-NEAR-MISS | 240 | 0.575 |   0.456 |   0.679 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-4b       | T2     | v1        | FAM-SAME-DIM  | 240 | 0.567 |   0.459 |   0.684 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-4b       | T2     | v1        | FAM-SAME-UNIT | 240 | 0.838 |   0.727 |   0.929 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-4b       | T2     | v1        | INV-BASE      | 240 | 0.775 |   0.594 |   0.94  |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-4b       | T2     | v1        | INV-LEX       | 240 | 0.808 |   0.693 |   0.904 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-4b       | T3     | v1        | FAM-NAMED     | 240 | 0.962 |   0.933 |   0.987 |    0.25  |              nan     |       nan     |       nan     |              nan     |
| qwen3-4b       | T3     | v1        | FAM-UNNAMED   | 240 | 0.858 |   0.787 |   0.924 |    0.25  |              nan     |       nan     |       nan     |              nan     |
| qwen3-4b       | T3     | v1        | INV-LEX       | 240 | 0.892 |   0.818 |   0.96  |    0.25  |              nan     |       nan     |       nan     |              nan     |
| qwen3-4b       | T4     | v1        | FAM-DIFF-FAR  | 240 | 0.858 |   0.793 |   0.913 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-4b       | T4     | v1        | FAM-DIFF-NEAR | 240 | 0.771 |   0.708 |   0.827 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-4b       | T4     | v1        | FAM-NEAR-MISS | 240 | 0.792 |   0.722 |   0.858 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-4b       | T4     | v1        | FAM-SAME-DIM  | 240 | 0.871 |   0.796 |   0.933 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-4b       | T4     | v1        | INV-LEX       | 240 | 0.754 |   0.549 |   0.971 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-4b       | T5     | v1        | FAM-FAR       | 240 | 0.546 |   0.464 |   0.639 |    0.333 |              nan     |       nan     |       nan     |              nan     |
| qwen3-4b       | T5     | v1        | FAM-NEAR      | 240 | 0.5   |   0.427 |   0.567 |    0.333 |              nan     |       nan     |       nan     |              nan     |
| qwen3-4b       | T5     | v1        | INV-LEX       | 240 | 0.471 |   0.332 |   0.618 |    0.333 |              nan     |       nan     |       nan     |              nan     |
| qwen3-4b-base  | T1     | v2        | FAM-NAMED     | 480 | 0.981 |   0.961 |   0.996 |    0.25  |                0.795 |         0.983 |         0.979 |                0.986 |
| qwen3-4b-base  | T1     | v2        | FAM-UNNAMED   | 480 | 0.85  |   0.767 |   0.913 |    0.25  |                0.617 |         0.827 |         0.879 |                0.851 |
| qwen3-4b-base  | T1     | v2        | INV-BASE      | 480 | 0.944 |   0.89  |   0.988 |    0.25  |                0.89  |         0.961 |         0.923 |                0.944 |
| qwen3-4b-base  | T1     | v2        | INV-BASE-TWIN | 480 | 0.879 |   0.783 |   0.962 |    0.25  |                0.832 |         0.923 |         0.828 |                0.892 |
| qwen3-4b-base  | T1     | v2        | INV-LEX       | 480 | 0.948 |   0.91  |   0.978 |    0.25  |                0.519 |         0.946 |         0.95  |                0.952 |
| qwen3-4b-base  | T1     | v2        | INV-LEX-TWIN  | 480 | 0.919 |   0.871 |   0.961 |    0.25  |                0.766 |         0.95  |         0.876 |                0.932 |
| qwen3-4b-base  | T2     | v1        | FAM-DIFF-FAR  | 240 | 0.588 |   0.465 |   0.716 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-4b-base  | T2     | v1        | FAM-NEAR-MISS | 240 | 0.604 |   0.481 |   0.715 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-4b-base  | T2     | v1        | FAM-SAME-DIM  | 240 | 0.538 |   0.411 |   0.681 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-4b-base  | T2     | v1        | FAM-SAME-UNIT | 240 | 0.692 |   0.508 |   0.849 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-4b-base  | T2     | v1        | INV-BASE      | 240 | 0.542 |   0.441 |   0.679 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-4b-base  | T2     | v1        | INV-LEX       | 240 | 0.688 |   0.533 |   0.84  |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-4b-base  | T3     | v1        | FAM-NAMED     | 240 | 0.912 |   0.874 |   0.949 |    0.25  |              nan     |       nan     |       nan     |              nan     |
| qwen3-4b-base  | T3     | v1        | FAM-UNNAMED   | 240 | 0.738 |   0.655 |   0.831 |    0.25  |              nan     |       nan     |       nan     |              nan     |
| qwen3-4b-base  | T3     | v1        | INV-LEX       | 240 | 0.833 |   0.784 |   0.879 |    0.25  |              nan     |       nan     |       nan     |              nan     |
| qwen3-4b-base  | T4     | v1        | FAM-DIFF-FAR  | 240 | 0.742 |   0.573 |   0.907 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-4b-base  | T4     | v1        | FAM-DIFF-NEAR | 240 | 0.696 |   0.542 |   0.827 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-4b-base  | T4     | v1        | FAM-NEAR-MISS | 240 | 0.771 |   0.633 |   0.878 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-4b-base  | T4     | v1        | FAM-SAME-DIM  | 240 | 0.783 |   0.6   |   0.907 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-4b-base  | T4     | v1        | INV-LEX       | 240 | 0.854 |   0.623 |   0.996 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-4b-base  | T5     | v1        | FAM-FAR       | 240 | 0.512 |   0.408 |   0.653 |    0.333 |              nan     |       nan     |       nan     |              nan     |
| qwen3-4b-base  | T5     | v1        | FAM-NEAR      | 240 | 0.45  |   0.274 |   0.58  |    0.333 |              nan     |       nan     |       nan     |              nan     |
| qwen3-4b-base  | T5     | v1        | INV-LEX       | 240 | 0.504 |   0.433 |   0.574 |    0.333 |              nan     |       nan     |       nan     |              nan     |
| qwen3-8b       | T1     | v2        | FAM-NAMED     | 480 | 0.688 |   0.613 |   0.763 |    0.25  |                0.879 |         0.643 |         0.753 |                0.708 |
| qwen3-8b       | T1     | v2        | FAM-UNNAMED   | 480 | 0.371 |   0.24  |   0.528 |    0.25  |                0.618 |         0.387 |         0.35  |                0.371 |
| qwen3-8b       | T1     | v2        | INV-BASE      | 480 | 0.79  |   0.623 |   0.941 |    0.25  |                0.933 |         0.807 |         0.769 |                0.786 |
| qwen3-8b       | T1     | v2        | INV-BASE-TWIN | 480 | 0.525 |   0.345 |   0.736 |    0.25  |                0.793 |         0.598 |         0.439 |                0.556 |
| qwen3-8b       | T1     | v2        | INV-LEX       | 480 | 0.64  |   0.555 |   0.719 |    0.25  |                0.602 |         0.663 |         0.607 |                0.643 |
| qwen3-8b       | T1     | v2        | INV-LEX-TWIN  | 480 | 0.596 |   0.512 |   0.675 |    0.25  |                0.765 |         0.581 |         0.617 |                0.618 |
| qwen3-8b       | T2     | v1        | FAM-DIFF-FAR  | 240 | 0.7   |   0.606 |   0.798 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-8b       | T2     | v1        | FAM-NEAR-MISS | 240 | 0.633 |   0.494 |   0.752 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-8b       | T2     | v1        | FAM-SAME-DIM  | 240 | 0.625 |   0.496 |   0.752 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-8b       | T2     | v1        | FAM-SAME-UNIT | 240 | 0.912 |   0.829 |   0.987 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-8b       | T2     | v1        | INV-BASE      | 240 | 0.825 |   0.642 |   0.992 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-8b       | T2     | v1        | INV-LEX       | 240 | 0.896 |   0.759 |   0.984 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-8b       | T3     | v1        | FAM-NAMED     | 240 | 0.462 |   0.218 |   0.689 |    0.25  |              nan     |       nan     |       nan     |              nan     |
| qwen3-8b       | T3     | v1        | FAM-UNNAMED   | 240 | 0.458 |   0.299 |   0.649 |    0.25  |              nan     |       nan     |       nan     |              nan     |
| qwen3-8b       | T3     | v1        | INV-LEX       | 240 | 0.292 |   0.226 |   0.361 |    0.25  |              nan     |       nan     |       nan     |              nan     |
| qwen3-8b       | T4     | v1        | FAM-DIFF-FAR  | 240 | 0.896 |   0.852 |   0.935 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-8b       | T4     | v1        | FAM-DIFF-NEAR | 240 | 0.75  |   0.642 |   0.854 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-8b       | T4     | v1        | FAM-NEAR-MISS | 240 | 0.796 |   0.704 |   0.879 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-8b       | T4     | v1        | FAM-SAME-DIM  | 240 | 0.846 |   0.782 |   0.906 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-8b       | T4     | v1        | INV-LEX       | 240 | 0.917 |   0.835 |   0.97  |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-8b       | T5     | v1        | FAM-FAR       | 240 | 0.4   |   0.308 |   0.496 |    0.333 |              nan     |       nan     |       nan     |              nan     |
| qwen3-8b       | T5     | v1        | FAM-NEAR      | 240 | 0.433 |   0.32  |   0.529 |    0.333 |              nan     |       nan     |       nan     |              nan     |
| qwen3-8b       | T5     | v1        | INV-LEX       | 240 | 0.375 |   0.297 |   0.45  |    0.333 |              nan     |       nan     |       nan     |              nan     |
| qwen3-8b-base  | T1     | v2        | FAM-NAMED     | 480 | 0.994 |   0.981 |   1     |    0.25  |                0.769 |         0.997 |         0.99  |                0.994 |
| qwen3-8b-base  | T1     | v2        | FAM-UNNAMED   | 480 | 0.838 |   0.738 |   0.916 |    0.25  |                0.64  |         0.808 |         0.874 |                0.843 |
| qwen3-8b-base  | T1     | v2        | INV-BASE      | 480 | 0.91  |   0.825 |   0.988 |    0.25  |                0.79  |         0.915 |         0.905 |                0.914 |
| qwen3-8b-base  | T1     | v2        | INV-BASE-TWIN | 480 | 0.935 |   0.868 |   0.989 |    0.25  |                0.878 |         0.969 |         0.896 |                0.954 |
| qwen3-8b-base  | T1     | v2        | INV-LEX       | 480 | 0.956 |   0.924 |   0.982 |    0.25  |                0.467 |         0.961 |         0.95  |                0.964 |
| qwen3-8b-base  | T1     | v2        | INV-LEX-TWIN  | 480 | 0.912 |   0.856 |   0.959 |    0.25  |                0.731 |         0.932 |         0.886 |                0.929 |
| qwen3-8b-base  | T2     | v1        | FAM-DIFF-FAR  | 240 | 0.638 |   0.52  |   0.752 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-8b-base  | T2     | v1        | FAM-NEAR-MISS | 240 | 0.654 |   0.515 |   0.757 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-8b-base  | T2     | v1        | FAM-SAME-DIM  | 240 | 0.558 |   0.434 |   0.691 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-8b-base  | T2     | v1        | FAM-SAME-UNIT | 240 | 0.762 |   0.583 |   0.906 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-8b-base  | T2     | v1        | INV-BASE      | 240 | 0.55  |   0.466 |   0.656 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-8b-base  | T2     | v1        | INV-LEX       | 240 | 0.579 |   0.436 |   0.688 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-8b-base  | T3     | v1        | FAM-NAMED     | 240 | 0.971 |   0.944 |   0.992 |    0.25  |              nan     |       nan     |       nan     |              nan     |
| qwen3-8b-base  | T3     | v1        | FAM-UNNAMED   | 240 | 0.758 |   0.7   |   0.813 |    0.25  |              nan     |       nan     |       nan     |              nan     |
| qwen3-8b-base  | T3     | v1        | INV-LEX       | 240 | 0.904 |   0.86  |   0.943 |    0.25  |              nan     |       nan     |       nan     |              nan     |
| qwen3-8b-base  | T4     | v1        | FAM-DIFF-FAR  | 240 | 0.921 |   0.855 |   0.986 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-8b-base  | T4     | v1        | FAM-DIFF-NEAR | 240 | 0.779 |   0.65  |   0.87  |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-8b-base  | T4     | v1        | FAM-NEAR-MISS | 240 | 0.796 |   0.703 |   0.877 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-8b-base  | T4     | v1        | FAM-SAME-DIM  | 240 | 0.883 |   0.817 |   0.941 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-8b-base  | T4     | v1        | INV-LEX       | 240 | 0.875 |   0.732 |   0.988 |    0.5   |              nan     |       nan     |       nan     |              nan     |
| qwen3-8b-base  | T5     | v1        | FAM-FAR       | 240 | 0.504 |   0.424 |   0.597 |    0.333 |              nan     |       nan     |       nan     |              nan     |
| qwen3-8b-base  | T5     | v1        | FAM-NEAR      | 240 | 0.471 |   0.356 |   0.58  |    0.333 |              nan     |       nan     |       nan     |              nan     |
| qwen3-8b-base  | T5     | v1        | INV-LEX       | 240 | 0.562 |   0.453 |   0.664 |    0.333 |              nan     |       nan     |       nan     |              nan     |


## T1 v1 vs v2 (prefix bug fixed in v2)

|                                     |    v1 |    v2 |
|:------------------------------------|------:|------:|
| ('gemma2-9b', 'FAM-NAMED')          | 0.933 | 0.983 |
| ('gemma2-9b', 'FAM-UNNAMED')        | 0.862 | 0.865 |
| ('gemma2-9b', 'INV-BASE')           | 0.896 | 0.904 |
| ('gemma2-9b', 'INV-BASE-TWIN')      | 0.875 | 0.902 |
| ('gemma2-9b', 'INV-LEX')            | 0.848 | 0.892 |
| ('gemma2-9b', 'INV-LEX-TWIN')       | 0.871 | 0.919 |
| ('olmo3-32b', 'FAM-NAMED')          | 0.935 | 0.983 |
| ('olmo3-32b', 'FAM-UNNAMED')        | 0.902 | 0.906 |
| ('olmo3-32b', 'INV-BASE')           | 0.91  | 0.912 |
| ('olmo3-32b', 'INV-BASE-TWIN')      | 0.919 | 0.938 |
| ('olmo3-32b', 'INV-LEX')            | 0.929 | 0.952 |
| ('olmo3-32b', 'INV-LEX-TWIN')       | 0.9   | 0.942 |
| ('olmo3-7b', 'FAM-NAMED')           | 0.923 | 0.969 |
| ('olmo3-7b', 'FAM-UNNAMED')         | 0.862 | 0.86  |
| ('olmo3-7b', 'INV-BASE')            | 0.881 | 0.879 |
| ('olmo3-7b', 'INV-BASE-TWIN')       | 0.881 | 0.9   |
| ('olmo3-7b', 'INV-LEX')             | 0.885 | 0.91  |
| ('olmo3-7b', 'INV-LEX-TWIN')        | 0.862 | 0.923 |
| ('qwen3-14b-base', 'FAM-NAMED')     | 0.938 | 0.981 |
| ('qwen3-14b-base', 'FAM-UNNAMED')   | 0.875 | 0.881 |
| ('qwen3-14b-base', 'INV-BASE')      | 0.908 | 0.935 |
| ('qwen3-14b-base', 'INV-BASE-TWIN') | 0.94  | 0.94  |
| ('qwen3-14b-base', 'INV-LEX')       | 0.933 | 0.965 |
| ('qwen3-14b-base', 'INV-LEX-TWIN')  | 0.894 | 0.931 |
| ('qwen3-4b', 'FAM-NAMED')           | 0.85  | 0.84  |
| ('qwen3-4b', 'FAM-UNNAMED')         | 0.621 | 0.635 |
| ('qwen3-4b', 'INV-BASE')            | 0.875 | 0.892 |
| ('qwen3-4b', 'INV-BASE-TWIN')       | 0.779 | 0.812 |
| ('qwen3-4b', 'INV-LEX')             | 0.817 | 0.844 |
| ('qwen3-4b', 'INV-LEX-TWIN')        | 0.721 | 0.769 |
| ('qwen3-4b-base', 'FAM-NAMED')      | 0.919 | 0.981 |
| ('qwen3-4b-base', 'FAM-UNNAMED')    | 0.86  | 0.85  |
| ('qwen3-4b-base', 'INV-BASE')       | 0.94  | 0.944 |
| ('qwen3-4b-base', 'INV-BASE-TWIN')  | 0.862 | 0.879 |
| ('qwen3-4b-base', 'INV-LEX')        | 0.923 | 0.948 |
| ('qwen3-4b-base', 'INV-LEX-TWIN')   | 0.865 | 0.919 |
| ('qwen3-8b', 'FAM-NAMED')           | 0.681 | 0.688 |
| ('qwen3-8b', 'FAM-UNNAMED')         | 0.39  | 0.371 |
| ('qwen3-8b', 'INV-BASE')            | 0.771 | 0.79  |
| ('qwen3-8b', 'INV-BASE-TWIN')       | 0.542 | 0.525 |
| ('qwen3-8b', 'INV-LEX')             | 0.617 | 0.64  |
| ('qwen3-8b', 'INV-LEX-TWIN')        | 0.59  | 0.596 |
| ('qwen3-8b-base', 'FAM-NAMED')      | 0.938 | 0.994 |
| ('qwen3-8b-base', 'FAM-UNNAMED')    | 0.854 | 0.838 |
| ('qwen3-8b-base', 'INV-BASE')       | 0.915 | 0.91  |
| ('qwen3-8b-base', 'INV-BASE-TWIN')  | 0.921 | 0.935 |
| ('qwen3-8b-base', 'INV-LEX')        | 0.935 | 0.956 |
| ('qwen3-8b-base', 'INV-LEX-TWIN')   | 0.888 | 0.912 |