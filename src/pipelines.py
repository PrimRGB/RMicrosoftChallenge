from sklearn.pipeline import make_union, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as nps

from src.transformation_utils.select_column import SelectColumns

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

def get_pipeline():
    make_pipeline(
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
                                    ])
                    ),
                    SelectColumns(["Census_ProcessorCoreCount"]),
                ),
                SimpleImputer(missing_values=[np.nan, -1], startegy="median")
            )
        )
    )
