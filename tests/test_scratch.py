# import pandas as pd

# def test_scratch():
#     """
#     This test parses working requirements to be required in this library since `poetry` was added after its completion
#     to manage/resolve dependencies.
#
#     Returns
#     -------
#     None
#     """
#     reqs_df = \
#         pd.read_csv(
#             filepath_or_buffer='/Users/ShaunPorwal/Documents/' \
#                                'GitHub/python_packages/dcurves/working_requirements.tsv',
#             sep='\t',
#             header=None
#         )
#
#     reqs_df[['package', 'working', 'latest']] = reqs_df.iloc[:,0].str.split(r'\s+', expand=True)
#     reqs_df.drop(columns=0, inplace=True)
#
#     # print('\n', reqs_df.to_string())

# def test_nparange_replace():
#     pass
# print(
#     [0 + i*0.01 for i in range(0,1)]
# )
