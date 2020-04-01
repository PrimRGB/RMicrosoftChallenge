from sklearn.pipeline import make_union, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.svm import SVC

from src.transformation_utils.select_columns import SelectColumns

features = [
    "Census_MDC2FormFactor",
    "Census_DeviceFamily",
    "Census_OEMNameIdentifier",
    "Census_OEMModelIdentifier",
    "Census_ProcessorCoreCount",
    "Census_ProcessorManufacturerIdentifier",
    "Census_ProcessorModelIdentifier",
    "Census_ProcessorClass",
    "Census_PrimaryDiskTotalCapacity",
    "Census_PrimaryDiskTypeName",
    "Census_SystemVolumeTotalCapacity",
    "Census_HasOpticalDiskDrive",
    "Census_TotalPhysicalRAM",
    "Census_ChassisTypeName",
    "Census_InternalPrimaryDiagonalDisplaySizeInInches",
    "Census_InternalPrimaryDisplayResolutionHorizontal",
    "Census_InternalPrimaryDisplayResolutionVertical",
    "Census_GenuineStateName",
    "Census_ActivationChannel",
    "Census_FirmwareManufacturerIdentifier",
    "Census_FirmwareVersionIdentifier",
    "Census_IsTouchEnabled",
    "Census_IsPenCapable",
    "Census_IsAlwaysOnAlwaysConnectedCapable",
    "Wdft_IsGamer"
]

target_column_name = "HasDetections"

# TODO: MAD - Mean Absolute Deviation
def get_pipeline():
    pipeline = make_pipeline(
        SelectColumns(features),
        make_union(
            make_pipeline(
                SelectColumns(["Census_MDC2FormFactor"]),
                OneHotEncoder()
            ),
            make_pipeline(
                make_union(
                    make_pipeline(
                        SelectColumns(["Census_PrimaryDiskTotalCapacity",
                                   "Census_SystemVolumeTotalCapacity",
                                   "Census_TotalPhysicalRAM"
                                    ]),
                        RobustScaler()
                    ),
                    SelectColumns(["Census_ProcessorCoreCount"]),
                ),
                SimpleImputer(missing_values=np.nan, strategy="median"),
                SimpleImputer(missing_values=-1, strategy="median")
            )
        ),
        SVC(C=1.0, kernel='rbf')
    )
    return pipeline
