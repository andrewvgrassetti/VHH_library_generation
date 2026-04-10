from vhh_library.utils import AMINO_ACIDS

IMGT_REGIONS = {
    "FR1":  (1,  25),
    "CDR1": (26, 35),
    "FR2":  (36, 49),
    "CDR2": (50, 58),
    "FR3":  (59, 96),
    "CDR3": (97, 110),
    "FR4":  (111, 128),
}

CDR_REGIONS = {"CDR1", "CDR2", "CDR3"}
FR_REGIONS  = {"FR1", "FR2", "FR3", "FR4"}


class VHHSequence:
    def __init__(self, sequence: str):
        self._sequence = sequence.upper().strip()
        self._validation_result = self.validate()
        self._imgt_numbered = self.imgt_number()

    def validate(self) -> dict:
        errors = []
        warnings = []
        seq = self._sequence

        if len(seq) < 110:
            errors.append(f"Sequence too short ({len(seq)} aa); minimum is 110 aa for a VHH.")
        elif len(seq) > 130:
            errors.append(f"Sequence too long ({len(seq)} aa); maximum is 130 aa for a VHH.")

        if seq and seq[0] not in ("E", "Q", "D"):
            warnings.append(f"Sequence starts with '{seq[0]}'; typical VHH starts with E, Q, or D.")

        if len(seq) >= 23 and seq[22] != "C":
            warnings.append(f"Position 23 (IMGT ~23) is '{seq[22]}'; expected Cys for canonical disulfide.")
        if len(seq) >= 104 and seq[103] != "C":
            warnings.append(f"Position 104 (IMGT ~104) is '{seq[103]}'; expected Cys for canonical disulfide.")

        for i, aa in enumerate(seq):
            if aa not in AMINO_ACIDS:
                errors.append(f"Non-standard amino acid '{aa}' at position {i+1}.")

        valid = len(errors) == 0
        return {"valid": valid, "errors": errors, "warnings": warnings}

    def imgt_number(self) -> dict:
        numbered = {}
        for i, aa in enumerate(self._sequence):
            imgt_pos = i + 1
            if imgt_pos > 128:
                break
            numbered[imgt_pos] = aa
        return numbered

    def get_regions(self) -> dict:
        regions = {}
        for region_name, (start, end) in IMGT_REGIONS.items():
            positions = [p for p in range(start, end + 1) if p in self._imgt_numbered]
            if positions:
                seq_str = "".join(self._imgt_numbered[p] for p in positions)
                regions[region_name] = (start, end, seq_str)
            else:
                regions[region_name] = (start, end, "")
        return regions

    def get_cdr_positions(self) -> set:
        cdr_pos = set()
        for region_name, (start, end) in IMGT_REGIONS.items():
            if region_name in CDR_REGIONS:
                for p in range(start, end + 1):
                    if p in self._imgt_numbered:
                        cdr_pos.add(p)
        return cdr_pos

    def get_framework_positions(self) -> set:
        fr_pos = set()
        for region_name, (start, end) in IMGT_REGIONS.items():
            if region_name in FR_REGIONS:
                for p in range(start, end + 1):
                    if p in self._imgt_numbered:
                        fr_pos.add(p)
        return fr_pos

    @property
    def sequence(self) -> str:
        return self._sequence

    @property
    def imgt_numbered(self) -> dict:
        return self._imgt_numbered

    @property
    def regions(self) -> dict:
        return self.get_regions()

    @property
    def cdr_positions(self) -> set:
        return self.get_cdr_positions()

    @property
    def framework_positions(self) -> set:
        return self.get_framework_positions()

    @property
    def validation_result(self) -> dict:
        return self._validation_result

    @property
    def length(self) -> int:
        return len(self._sequence)
