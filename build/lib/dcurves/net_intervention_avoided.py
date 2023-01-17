import pandas as pd
import beartype

@beartype
def net_intervention_avoided(
        after_dca_df: pd.DataFrame,
        nper: int = 100
):

    all_records = after_dca_df[after_dca_df['variable'] == 'all']
    all_records = all_records[['threshold', 'net_benefit']]
    all_records = all_records.rename(columns={'net_benefit': 'net_benefit_all'})

    merged_after_dca_df = after_dca_df.merge(all_records, on='threshold')

    merged_after_dca_df['net_intervention_avoided'] = (merged_after_dca_df['net_benefit']
                                             - merged_after_dca_df['net_benefit_all']) \
                                            / (merged_after_dca_df['threshold']
                                               / (1 - merged_after_dca_df['threshold'])) * nper

    return merged_after_dca_df


net_intervention_avoided.__doc__ = """

    |

    Calculate net interventions avoided after performing decision curve analysis

    |

    Examples
    ________

    >>> df_binary = dcurves.load_test_data.load_binary_df()

    >>> after_dca_df = dcurves.dca(
    ...     data = df_binary,
    ...     outcome = 'cancer',
    ...     predictors = ['famhistory']
    ... )

    >>> after_net_intervention_avoided_df = dcurves.net_intervention_avoided(
    ... after_dca_df = after_dca_df,
    ... nper = 100
    ...)

    |

    Parameters
    __________
    after_dca_df : pd.DataFrame
        dataframe outputted by dca function in the dcurves library
    nper : int
        number to report net interventions per （Defaults to 100)

    Return
    ______
    merged_after_dca_df: pd.DataFrame
        dataframe with calculated net_intervention_avoided field joined to the inputted after_dca_df

    """