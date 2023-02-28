import pandas as pd


def time_series_windowing(
    df: pd.DataFrame,
    feature_column_name: str,
    window_size: int,
    reverse_direction: bool = False,
) -> pd.DataFrame:
    """묶음 학습을 위해 시계열 데이터를 windowing

    Args:
        df (pd.DataFrame): 데이터레임
        feature_column_name (str): feature로 쓸 컬럼 이름
        window_size (int): 시계열 데이터 묶음의 단위
        reverse_direction (bool): 시계열 데이터 묶음을 위한 진행 방향, True면 역방향, False면 정방향

    Returns:
        df: windowing 전처리가 완료된 데이터 프레임
    """
    list_shift = [df[feature_column_name]]
    column_names = ["shift_" + str(i) for i in range(1, window_size + 1)]

    if reverse_direction:
        column_names.insert(0, "target")
    else:
        column_names.append("target")

    for s in range(1, window_size + 1):
        if reverse_direction:
            shift_index = s
        else:
            shift_index = 0 - s

        list_shift.append(df[feature_column_name].shift(shift_index))

    df = pd.concat(list_shift, axis=1)
    df.columns = column_names
    df = df.dropna().reset_index(drop=True)
    return df
