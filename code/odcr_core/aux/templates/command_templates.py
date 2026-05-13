"""Registered command templates for ODCR auxiliary runtime workflows."""

RUNTIME_COMMAND_TEMPLATES = {
    "bridge_validate": "./odcr runtime bridge validate-only",
    "bridge_marker": "./odcr runtime bridge marker-probe",
    "bridge_cuda": "./odcr runtime bridge cuda-probe",
    "step3_bounded_probe": "./odcr runtime probe --stage step3 --task 2 --profile csb_odcr_full_safe --bounded",
}
