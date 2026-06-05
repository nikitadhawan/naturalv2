"""Representation of Clinical Trial Data from ClinicalTrials.gov."""

import json
import logging
import multiprocessing as mp
import os
from enum import Enum
from functools import partial
from typing import Any, List, Optional

import requests
from pydantic import BaseModel, Field
from tqdm import tqdm


logger = logging.getLogger(__name__)

# ruff: noqa


# -----------------------------------------------------------------------------
# Types and Enums
# -----------------------------------------------------------------------------
class StrEnum(str, Enum):
    """Base class for str enums to be handled properly by pydantic."""

    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return name


class Status(StrEnum):
    ACTIVE_NOT_RECRUITING = "ACTIVE_NOT_RECRUITING"  # Active, not recruiting
    COMPLETED = "COMPLETED"  # Completed
    ENROLLING_BY_INVITATION = "ENROLLING_BY_INVITATION"  # Enrolling by invitation
    NOT_YET_RECRUITING = "NOT_YET_RECRUITING"  # Not yet recruiting
    RECRUITING = "RECRUITING"  # Recruiting
    SUSPENDED = "SUSPENDED"  # Suspended
    TERMINATED = "TERMINATED"  # Terminated
    WITHDRAWN = "WITHDRAWN"  # Withdrawn
    AVAILABLE = "AVAILABLE"  # Available
    NO_LONGER_AVAILABLE = "NO_LONGER_AVAILABLE"  # No longer available
    TEMPORARILY_NOT_AVAILABLE = "TEMPORARILY_NOT_AVAILABLE"  # Temporarily not available
    APPROVED_FOR_MARKETING = "APPROVED_FOR_MARKETING"  # Approved for marketing
    WITHHELD = "WITHHELD"  # Withheld
    UNKNOWN = "UNKNOWN"  # Unknown status


class StudyType(StrEnum):
    EXPANDED_ACCESS = "EXPANDED_ACCESS"  # Expanded Access
    INTERVENTIONAL = "INTERVENTIONAL"  # Interventional
    OBSERVATIONAL = "OBSERVATIONAL"  # Observational


class Phase(StrEnum):
    NA = "NA"  # Not Applicable
    EARLY_PHASE1 = "EARLY_PHASE1"  # Early Phase 1
    PHASE1 = "PHASE1"  # Phase 1
    PHASE2 = "PHASE2"  # Phase 2
    PHASE3 = "PHASE3"  # Phase 3
    PHASE4 = "PHASE4"  # Phase 4


class Sex(StrEnum):
    FEMALE = "FEMALE"  # Female
    MALE = "MALE"  # Male
    ALL = "ALL"  # All


class StandardAge(StrEnum):
    CHILD = "CHILD"  # Child
    ADULT = "ADULT"  # Adult
    OLDER_ADULT = "OLDER_ADULT"  # Older Adult


class SamplingMethod(StrEnum):
    PROBABILITY_SAMPLE = "PROBABILITY_SAMPLE"  # Probability Sample
    NON_PROBABILITY_SAMPLE = "NON_PROBABILITY_SAMPLE"  # Non-Probability Sample


class IpdSharing(StrEnum):
    YES = "YES"  # Yes
    NO = "NO"  # No
    UNDECIDED = "UNDECIDED"  # Undecided


class IpdSharingInfoType(StrEnum):
    STUDY_PROTOCOL = "STUDY_PROTOCOL"  # Study Protocol
    SAP = "SAP"  # Statistical Analysis Plan (SAP)
    ICF = "ICF"  # Informed Consent Form (ICF)
    CSR = "CSR"  # Clinical Study Report (CSR)
    ANALYTIC_CODE = "ANALYTIC_CODE"  # Analytic Code


class OrgStudyIdType(StrEnum):
    NIH = "NIH"  # U.S. NIH Grant/Contract
    FDA = "FDA"  # U.S. FDA Grant/Contract
    VA = "VA"  # U.S. VA Grant/Contract
    CDC = "CDC"  # U.S. CDC Grant/Contract
    AHRQ = "AHRQ"  # U.S. AHRQ Grant/Contract
    SAMHSA = "SAMHSA"  # U.S. SAMHSA Grant/Contract


class SecondaryIdType(StrEnum):
    NIH = "NIH"  # U.S. NIH Grant/Contract
    FDA = "FDA"  # U.S. FDA Grant/Contract
    VA = "VA"  # U.S. VA Grant/Contract
    CDC = "CDC"  # U.S. CDC Grant/Contract
    AHRQ = "AHRQ"  # U.S. AHRQ Grant/Contract
    SAMHSA = "SAMHSA"  # U.S. SAMHSA Grant/Contract
    OTHER_GRANT = "OTHER_GRANT"  # Other Grant/Funding Number
    EUDRACT_NUMBER = "EUDRACT_NUMBER"  # EudraCT Number
    CTIS = "CTIS"  # EU Trial (CTIS) Number
    REGISTRY = "REGISTRY"  # Registry Identifier
    OTHER = "OTHER"  # Other Identifier


class AgencyClass(StrEnum):
    NIH = "NIH"  # NIH
    FED = "FED"  # FED
    OTHER_GOV = "OTHER_GOV"  # OTHER_GOV
    INDIV = "INDIV"  # INDIV
    INDUSTRY = "INDUSTRY"  # INDUSTRY
    NETWORK = "NETWORK"  # NETWORK
    AMBIG = "AMBIG"  # AMBIG
    OTHER = "OTHER"  # OTHER
    UNKNOWN = "UNKNOWN"  # UNKNOWN


class ExpandedAccessStatus(StrEnum):
    AVAILABLE = "AVAILABLE"  # Available
    NO_LONGER_AVAILABLE = "NO_LONGER_AVAILABLE"  # No longer available
    TEMPORARILY_NOT_AVAILABLE = "TEMPORARILY_NOT_AVAILABLE"  # Temporarily not available
    APPROVED_FOR_MARKETING = "APPROVED_FOR_MARKETING"  # Approved for marketing


class DateType(StrEnum):
    ACTUAL = "ACTUAL"  # Actual
    ESTIMATED = "ESTIMATED"  # Estimated


class ResponsiblePartyType(StrEnum):
    SPONSOR = "SPONSOR"  # Sponsor
    PRINCIPAL_INVESTIGATOR = "PRINCIPAL_INVESTIGATOR"  # Principal Investigator
    SPONSOR_INVESTIGATOR = "SPONSOR_INVESTIGATOR"  # Sponsor-Investigator


class DesignAllocation(StrEnum):
    RANDOMIZED = "RANDOMIZED"  # Randomized
    NON_RANDOMIZED = "NON_RANDOMIZED"  # Non-Randomized
    NA = "NA"  # N/A


class InterventionalAssignment(StrEnum):
    SINGLE_GROUP = "SINGLE_GROUP"  # Single Group Assignment
    PARALLEL = "PARALLEL"  # Parallel Assignment
    CROSSOVER = "CROSSOVER"  # Crossover Assignment
    FACTORIAL = "FACTORIAL"  # Factorial Assignment
    SEQUENTIAL = "SEQUENTIAL"  # Sequential Assignment


class PrimaryPurpose(StrEnum):
    TREATMENT = "TREATMENT"  # Treatment
    PREVENTION = "PREVENTION"  # Prevention
    DIAGNOSTIC = "DIAGNOSTIC"  # Diagnostic
    ECT = "ECT"  # Educational/Counseling/Training
    SUPPORTIVE_CARE = "SUPPORTIVE_CARE"  # Supportive Care
    SCREENING = "SCREENING"  # Screening
    HEALTH_SERVICES_RESEARCH = "HEALTH_SERVICES_RESEARCH"  # Health Services Research
    BASIC_SCIENCE = "BASIC_SCIENCE"  # Basic Science
    DEVICE_FEASIBILITY = "DEVICE_FEASIBILITY"  # Device Feasibility
    OTHER = "OTHER"  # Other


class ObservationalModel(StrEnum):
    COHORT = "COHORT"  # Cohort
    CASE_CONTROL = "CASE_CONTROL"  # Case-Control
    CASE_ONLY = "CASE_ONLY"  # Case-Only
    CASE_CROSSOVER = "CASE_CROSSOVER"  # Case-Crossover
    ECOLOGIC_OR_COMMUNITY = "ECOLOGIC_OR_COMMUNITY"  # Ecologic or Community
    FAMILY_BASED = "FAMILY_BASED"  # Family-Based
    DEFINED_POPULATION = "DEFINED_POPULATION"  # Defined Population
    NATURAL_HISTORY = "NATURAL_HISTORY"  # Natural History
    OTHER = "OTHER"  # Other


class DesignTimePerspective(StrEnum):
    RETROSPECTIVE = "RETROSPECTIVE"  # Retrospective
    PROSPECTIVE = "PROSPECTIVE"  # Prospective
    CROSS_SECTIONAL = "CROSS_SECTIONAL"  # Cross-Sectional
    OTHER = "OTHER"  # Other


class BioSpecRetention(StrEnum):
    NONE_RETAINED = "NONE_RETAINED"  # None Retained
    SAMPLES_WITH_DNA = "SAMPLES_WITH_DNA"  # Samples With DNA
    SAMPLES_WITHOUT_DNA = "SAMPLES_WITHOUT_DNA"  # Samples Without DNA


class EnrollmentType(StrEnum):
    ACTUAL = "ACTUAL"  # Actual
    ESTIMATED = "ESTIMATED"  # Estimated


class ArmGroupType(StrEnum):
    EXPERIMENTAL = "EXPERIMENTAL"  # Experimental
    ACTIVE_COMPARATOR = "ACTIVE_COMPARATOR"  # Active Comparator
    PLACEBO_COMPARATOR = "PLACEBO_COMPARATOR"  # Placebo Comparator
    SHAM_COMPARATOR = "SHAM_COMPARATOR"  # Sham Comparator
    NO_INTERVENTION = "NO_INTERVENTION"  # No Intervention
    OTHER = "OTHER"  # Other


class InterventionType(StrEnum):
    BEHAVIORAL = "BEHAVIORAL"  # Behavioral
    BIOLOGICAL = "BIOLOGICAL"  # Biological
    COMBINATION_PRODUCT = "COMBINATION_PRODUCT"  # Combination Product
    DEVICE = "DEVICE"  # Device
    DIAGNOSTIC_TEST = "DIAGNOSTIC_TEST"  # Diagnostic Test
    DIETARY_SUPPLEMENT = "DIETARY_SUPPLEMENT"  # Dietary Supplement
    DRUG = "DRUG"  # Drug
    GENETIC = "GENETIC"  # Genetic
    PROCEDURE = "PROCEDURE"  # Procedure
    RADIATION = "RADIATION"  # Radiation
    OTHER = "OTHER"  # Other


class ContactRole(StrEnum):
    STUDY_CHAIR = "STUDY_CHAIR"  # Study Chair
    STUDY_DIRECTOR = "STUDY_DIRECTOR"  # Study Director
    PRINCIPAL_INVESTIGATOR = "PRINCIPAL_INVESTIGATOR"  # Principal Investigator
    SUB_INVESTIGATOR = "SUB_INVESTIGATOR"  # Sub-Investigator
    CONTACT = "CONTACT"  # Contact


class OfficialRole(StrEnum):
    STUDY_CHAIR = "STUDY_CHAIR"  # Study Chair
    STUDY_DIRECTOR = "STUDY_DIRECTOR"  # Study Director
    PRINCIPAL_INVESTIGATOR = "PRINCIPAL_INVESTIGATOR"  # Principal Investigator
    SUB_INVESTIGATOR = "SUB_INVESTIGATOR"  # Sub-Investigator


class RecruitmentStatus(StrEnum):
    ACTIVE_NOT_RECRUITING = "ACTIVE_NOT_RECRUITING"  # Active, not recruiting
    COMPLETED = "COMPLETED"  # Completed
    ENROLLING_BY_INVITATION = "ENROLLING_BY_INVITATION"  # Enrolling by invitation
    NOT_YET_RECRUITING = "NOT_YET_RECRUITING"  # Not yet recruiting
    RECRUITING = "RECRUITING"  # Recruiting
    SUSPENDED = "SUSPENDED"  # Suspended
    TERMINATED = "TERMINATED"  # Terminated
    WITHDRAWN = "WITHDRAWN"  # Withdrawn
    AVAILABLE = "AVAILABLE"  # Available


class ReferenceType(StrEnum):
    BACKGROUND = "BACKGROUND"  # background
    RESULT = "RESULT"  # result
    DERIVED = "DERIVED"  # derived


class MeasureParam(StrEnum):
    GEOMETRIC_MEAN = "GEOMETRIC_MEAN"  # Geometric Mean
    GEOMETRIC_LEAST_SQUARES_MEAN = (
        "GEOMETRIC_LEAST_SQUARES_MEAN"  # Geometric Least Squares Mean
    )
    LEAST_SQUARES_MEAN = "LEAST_SQUARES_MEAN"  # Least Squares Mean
    LOG_MEAN = "LOG_MEAN"  # Log Mean
    MEAN = "MEAN"  # Mean
    MEDIAN = "MEDIAN"  # Median
    NUMBER = "NUMBER"  # Number
    COUNT_OF_PARTICIPANTS = "COUNT_OF_PARTICIPANTS"  # Count of Participants
    COUNT_OF_UNITS = "COUNT_OF_UNITS"  # Count of Units


class MeasureDispersionType(StrEnum):
    NA = "NA"  # Not Applicable
    STANDARD_DEVIATION = "STANDARD_DEVIATION"  # Standard Deviation
    STANDARD_ERROR = "STANDARD_ERROR"  # Standard Error
    INTER_QUARTILE_RANGE = "INTER_QUARTILE_RANGE"  # Inter-Quartile Range
    FULL_RANGE = "FULL_RANGE"  # Full Range
    CONFIDENCE_80 = "CONFIDENCE_80"  # 80% Confidence Interval
    CONFIDENCE_90 = "CONFIDENCE_90"  # 90% Confidence Interval
    CONFIDENCE_95 = "CONFIDENCE_95"  # 95% Confidence Interval
    CONFIDENCE_975 = "CONFIDENCE_975"  # 97.5% Confidence Interval
    CONFIDENCE_99 = "CONFIDENCE_99"  # 99% Confidence Interval
    CONFIDENCE_OTHER = "CONFIDENCE_OTHER"  # Other Confidence Interval Level
    GEOMETRIC_COEFFICIENT = (
        "GEOMETRIC_COEFFICIENT"  # Geometric Coefficient of Variation
    )


class OutcomeMeasureType(StrEnum):
    PRIMARY = "PRIMARY"  # Primary
    SECONDARY = "SECONDARY"  # Secondary
    OTHER_PRE_SPECIFIED = "OTHER_PRE_SPECIFIED"  # Other Pre-specified
    POST_HOC = "POST_HOC"  # Post-Hoc


class ReportingStatus(StrEnum):
    NOT_POSTED = "NOT_POSTED"  # Not Posted
    POSTED = "POSTED"  # Posted


class EventAssessment(StrEnum):
    NON_SYSTEMATIC_ASSESSMENT = "NON_SYSTEMATIC_ASSESSMENT"  # Non-systematic Assessment
    SYSTEMATIC_ASSESSMENT = "SYSTEMATIC_ASSESSMENT"  # Systematic Assessment


class AgreementRestrictionType(StrEnum):
    LTE60 = "LTE60"  # LTE60
    GT60 = "GT60"  # GT60
    OTHER = "OTHER"  # OTHER


class BrowseLeafRelevance(StrEnum):
    LOW = "LOW"  # low
    HIGH = "HIGH"  # high


class DesignMasking(StrEnum):
    NONE = "NONE"  # None (Open Label)
    SINGLE = "SINGLE"  # Single
    DOUBLE = "DOUBLE"  # Double
    TRIPLE = "TRIPLE"  # Triple
    QUADRUPLE = "QUADRUPLE"  # Quadruple


class WhoMasked(StrEnum):
    PARTICIPANT = "PARTICIPANT"  # Participant
    CARE_PROVIDER = "CARE_PROVIDER"  # Care Provider
    INVESTIGATOR = "INVESTIGATOR"  # Investigator
    OUTCOMES_ASSESSOR = "OUTCOMES_ASSESSOR"  # Outcomes Assessor


class AnalysisDispersionType(StrEnum):
    STANDARD_DEVIATION = "STANDARD_DEVIATION"  # Standard Deviation
    STANDARD_ERROR_OF_MEAN = "STANDARD_ERROR_OF_MEAN"  # Standard Error of the Mean


class ConfidenceIntervalNumSides(StrEnum):
    ONE_SIDED = "ONE_SIDED"  # 1-Sided
    TWO_SIDED = "TWO_SIDED"  # 2-Sided


class NonInferiorityType(StrEnum):
    SUPERIORITY = "SUPERIORITY"  # Superiority
    NON_INFERIORITY = "NON_INFERIORITY"  # Non-Inferiority
    EQUIVALENCE = "EQUIVALENCE"  # Equivalence
    OTHER = "OTHER"  # Other
    NON_INFERIORITY_OR_EQUIVALENCE = (
        "NON_INFERIORITY_OR_EQUIVALENCE"  # Non-Inferiority or Equivalence
    )
    SUPERIORITY_OR_OTHER = "SUPERIORITY_OR_OTHER"  # Superiority or Other
    NON_INFERIORITY_OR_EQUIVALENCE_LEGACY = "NON_INFERIORITY_OR_EQUIVALENCE_LEGACY"  # Non-Inferiority or Equivalence (legacy)
    SUPERIORITY_OR_OTHER_LEGACY = (
        "SUPERIORITY_OR_OTHER_LEGACY"  # Superiority or Other (legacy)
    )


class UnpostedEventType(StrEnum):
    RESET = "RESET"  # Reset
    RELEASE = "RELEASE"  # Release
    UNRELEASE = "UNRELEASE"  # Unrelease


class ViolationEventType(StrEnum):
    VIOLATION_IDENTIFIED = "VIOLATION_IDENTIFIED"  # Violation Identified by FDA
    CORRECTION_CONFIRMED = "CORRECTION_CONFIRMED"  # Correction Confirmed by FDA
    PENALTY_IMPOSED = "PENALTY_IMPOSED"  # Penalty Imposed by FDA
    ISSUES_IN_LETTER_ADDRESSED_CONFIRMED = "ISSUES_IN_LETTER_ADDRESSED_CONFIRMED"  # Issues in letter addressed; confirmed by FDA.


class GeoPoint(BaseModel):
    lat: float
    lon: float


# -----------------------------------------------------------------------------
# Study Data Structure
# -----------------------------------------------------------------------------
class OrgStudyIdInfo(BaseModel):
    id: str
    type: Optional[OrgStudyIdType] = None
    link: Optional[str] = None


class SecondaryIdInfo(BaseModel):
    id: Optional[str] = None
    type: Optional[SecondaryIdType] = None
    domain: Optional[str] = None
    link: Optional[str] = None


class Organization(BaseModel):
    fullName: str
    class_: Optional[AgencyClass] = Field(None, alias="class")


class IdentificationModule(BaseModel):
    nctId: str
    briefTitle: str
    nctIdAliases: Optional[List[str]] = None
    orgStudyIdInfo: Optional[OrgStudyIdInfo] = None
    secondaryIdInfos: Optional[List[SecondaryIdInfo]] = None
    officialTitle: Optional[str] = None
    acronym: Optional[str] = None
    organization: Organization


class ExpandedAccessInfo(BaseModel):
    hasExpandedAccess: Optional[bool] = None
    nctId: Optional[str] = None
    statusForNctId: Optional[ExpandedAccessStatus] = None


class PartialDateStruct(BaseModel):
    date: Optional[str] = None
    type: Optional[DateType] = None


class DateStruct(BaseModel):
    date: Optional[str] = None
    type: Optional[DateType] = None


class StatusModule(BaseModel):
    statusVerifiedDate: str
    overallStatus: str
    primaryCompletionDateStruct: Optional[PartialDateStruct] = None
    lastKnownStatus: Optional[Status] = None
    delayedPosting: Optional[bool] = None
    whyStopped: Optional[str] = None
    expandedAccessInfo: Optional[ExpandedAccessInfo] = None
    startDateStruct: Optional[PartialDateStruct] = None
    completionDateStruct: Optional[PartialDateStruct] = None
    studyFirstSubmitDate: Optional[str] = None
    studyFirstSubmitQcDate: Optional[str] = None
    studyFirstPostDateStruct: Optional[DateStruct] = None
    studyFirstPostYear: Optional[int] = None
    resultsWaived: Optional[bool] = None
    resultsFirstSubmitDate: Optional[str] = None
    resultsFirstSubmitQcDate: Optional[str] = None
    resultsFirstPostDateStruct: Optional[DateStruct] = None
    dispFirstSubmitDate: Optional[str] = None
    dispFirstSubmitYear: Optional[int] = None
    dispFirstSubmitQcDate: Optional[str] = None
    dispFirstPostDateStruct: Optional[DateStruct] = None
    lastUpdateSubmitDate: Optional[str] = None
    lastUpdatePostDateStruct: Optional[DateStruct] = None


class ResponsibleParty(BaseModel):
    type: Optional[ResponsiblePartyType] = None
    investigatorFullName: Optional[str] = None
    investigatorTitle: Optional[str] = None
    investigatorAffiliation: Optional[str] = None
    oldNameTitle: Optional[str] = None
    oldOrganization: Optional[str] = None


class Sponsor(BaseModel):
    name: str
    class_: Optional[AgencyClass] = Field(None, alias="class")


class SponsorCollaboratorsModule(BaseModel):
    responsibleParty: Optional[ResponsibleParty] = None
    leadSponsor: Optional[Sponsor] = None
    collaborators: Optional[List[Sponsor]] = None


class OversightModule(BaseModel):
    oversightHasDmc: Optional[bool] = None
    isFdaRegulatedDrug: Optional[bool] = None
    isFdaRegulatedDevice: Optional[bool] = None
    isUnapprovedDevice: Optional[bool] = None
    isPpsd: Optional[bool] = None
    isUsExport: Optional[bool] = None
    fdaaa801Violation: Optional[bool] = None


class DescriptionModule(BaseModel):
    briefSummary: str
    detailedDescription: Optional[str] = None


class ConditionsModule(BaseModel):
    conditions: List[str]
    keywords: Optional[List[str]] = None


class ExpandedAccessTypes(BaseModel):
    individual: Optional[bool] = None
    intermediate: Optional[bool] = None
    treatment: Optional[bool] = None


class MaskingBlock(BaseModel):
    masking: Optional[DesignMasking] = None
    maskingDescription: Optional[str] = None
    whoMasked: Optional[List[WhoMasked]] = None


class DesignInfo(BaseModel):
    allocation: Optional[DesignAllocation] = None
    interventionModel: Optional[InterventionalAssignment] = None
    interventionModelDescription: Optional[str] = None
    primaryPurpose: Optional[PrimaryPurpose] = None
    observationalModel: Optional[ObservationalModel] = None
    timePerspective: Optional[DesignTimePerspective] = None
    maskingInfo: Optional[MaskingBlock] = None


class BioSpec(BaseModel):
    retention: Optional[BioSpecRetention] = None
    description: Optional[str] = None


class EnrollmentInfo(BaseModel):
    count: int
    type: EnrollmentType


class DesignModule(BaseModel):
    studyType: StudyType
    phases: Optional[List[Phase]] = None
    nPtrsToThisExpAccNctId: Optional[int] = None
    expandedAccessTypes: Optional[ExpandedAccessTypes] = None
    patientRegistry: Optional[bool] = None
    targetDuration: Optional[str] = None
    designInfo: Optional[DesignInfo] = None
    bioSpec: Optional[BioSpec] = None
    enrollmentInfo: Optional[EnrollmentInfo] = None


class ArmGroup(BaseModel):
    label: str
    type: Optional[ArmGroupType] = None
    description: Optional[str] = None
    interventionNames: Optional[List[str]] = None


class Intervention(BaseModel):
    type: InterventionType
    name: str
    description: Optional[str] = None
    armGroupLabels: Optional[List[str]] = None
    otherNames: Optional[List[str]] = None


class ArmsInterventionsModule(BaseModel):
    armGroups: Optional[List[ArmGroup]] = None
    interventions: Optional[List[Intervention]] = None


class Outcome(BaseModel):
    measure: str
    timeFrame: Optional[str] = None
    description: Optional[str] = None


class OutcomesModule(BaseModel):
    primaryOutcomes: Optional[List[Outcome]] = None
    secondaryOutcomes: Optional[List[Outcome]] = None
    otherOutcomes: Optional[List[Outcome]] = None


class EligibilityModule(BaseModel):
    eligibilityCriteria: Optional[str] = None
    sex: Optional[Sex] = None
    minimumAge: Optional[str] = None
    maximumAge: Optional[str] = None
    healthyVolunteers: Optional[bool] = None
    genderBased: Optional[bool] = None
    genderDescription: Optional[str] = None
    stdAges: Optional[List[StandardAge]] = None
    studyPopulation: Optional[str] = None
    samplingMethod: Optional[SamplingMethod] = None


class Contact(BaseModel):
    name: Optional[str] = None
    role: Optional[ContactRole] = None
    phone: Optional[str] = None
    phoneExt: Optional[str] = None
    email: Optional[str] = None


class Official(BaseModel):
    name: str
    affiliation: Optional[str] = None
    role: Optional[str] = None


class Location(BaseModel):
    city: Optional[str] = None
    country: Optional[str] = None
    facility: Optional[str] = None
    state: Optional[str] = None
    zip: Optional[str] = None
    status: Optional[RecruitmentStatus] = None
    contacts: Optional[List[Contact]] = None
    geoPoint: Optional[GeoPoint] = None


class ContactsLocationsModule(BaseModel):
    centralContacts: Optional[List[Contact]] = None
    overallOfficials: Optional[List[Official]] = None
    locations: Optional[List[Location]] = None


class Retraction(BaseModel):
    pmid: str
    source: str


class Reference(BaseModel):
    pmid: Optional[str] = None
    type: Optional[ReferenceType] = None
    citation: Optional[str] = None
    retractions: Optional[List[Retraction]] = None


class SeeAlsoLink(BaseModel):
    label: Optional[str] = None
    url: str


class AvailIpd(BaseModel):
    id: Optional[str] = None
    type: Optional[str] = None
    url: Optional[str] = None
    comment: Optional[str] = None


class IpdSharingStatementModule(BaseModel):
    ipdSharing: Optional[IpdSharing] = None
    description: Optional[str] = None
    infoTypes: Optional[List[IpdSharingInfoType]] = None
    timeFrame: Optional[str] = None
    accessCriteria: Optional[str] = None
    url: Optional[str] = None


class ReferencesModule(BaseModel):
    references: Optional[List[Reference]] = None
    retractionsAllRefs: Optional[int] = None  # Added
    seeAlsoLinks: Optional[List[SeeAlsoLink]] = None
    availIpds: Optional[List[AvailIpd]] = None
    ipdSharingStatementModule: Optional[IpdSharingStatementModule] = None


class ProtocolSection(BaseModel):
    identificationModule: IdentificationModule
    statusModule: StatusModule
    sponsorCollaboratorsModule: Optional[SponsorCollaboratorsModule] = None
    oversightModule: Optional[OversightModule] = None
    descriptionModule: Optional[DescriptionModule] = None
    conditionsModule: Optional[ConditionsModule] = None
    designModule: Optional[DesignModule] = None
    armsInterventionsModule: Optional[ArmsInterventionsModule] = None
    outcomesModule: Optional[OutcomesModule] = None
    eligibilityModule: Optional[EligibilityModule] = None
    contactsLocationsModule: Optional[ContactsLocationsModule] = None
    referencesModule: Optional[ReferencesModule] = None


class FlowGroup(BaseModel):
    id: str
    title: Optional[str] = None
    description: Optional[str] = None


class FlowStats(BaseModel):
    groupId: str
    comment: Optional[str] = None
    numSubjects: Optional[str] = None
    numUnits: Optional[str] = None


class FlowMilestone(BaseModel):
    type: str
    comment: Optional[str] = None
    achievements: Optional[List[FlowStats]] = None


class DropWithdraw(BaseModel):
    type: Optional[str] = None
    comment: Optional[str] = None
    reasons: Optional[List[FlowStats]] = None


class FlowPeriod(BaseModel):
    title: str
    milestones: Optional[List[FlowMilestone]]
    dropWithdraws: Optional[List[DropWithdraw]] = None


class ParticipantFlowModule(BaseModel):
    preAssignmentDetails: Optional[str] = None
    recruitmentDetails: Optional[str] = None
    typeUnitsAnalyzed: Optional[str] = None
    groups: Optional[List[FlowGroup]] = None
    periods: Optional[List[FlowPeriod]] = None


class DenomCount(BaseModel):
    groupId: str
    value: str

    def model_post_init(self, __context: Any) -> None:
        """Post init hook to handle value normalization."""
        self.value = self.value.replace(",", "")

        if self.value in ["NA", "N/A"]:
            self.value = "None"


class Denom(BaseModel):
    units: Optional[str] = None
    counts: Optional[List[DenomCount]] = None


class Measurement(BaseModel):
    groupId: str
    value: str
    spread: Optional[str] = None
    lowerLimit: Optional[str] = None
    upperLimit: Optional[str] = None
    comment: Optional[str] = None

    def model_post_init(self, __context: Any) -> None:
        """Post init hook to handle value normalization."""
        self.value = self.value.replace(",", "")

        if self.value in ["NA", "N/A"]:
            self.value = "None"


class MeasureCategory(BaseModel):
    title: Optional[str] = None
    measurements: Optional[List[Measurement]] = None


class MeasureClass(BaseModel):
    title: Optional[str] = None
    denoms: Optional[List[Denom]] = None
    categories: Optional[List[MeasureCategory]] = None


class BaselineMeasure(BaseModel):
    title: str
    paramType: Optional[MeasureParam] = None
    dispersionType: Optional[MeasureDispersionType] = None
    description: Optional[str] = None
    populationDescription: Optional[str] = None
    unitOfMeasure: Optional[str] = None
    calculatePct: Optional[bool] = None
    denomUnitsSelected: Optional[str] = None
    denoms: Optional[List[Denom]] = None
    classes: Optional[List[MeasureClass]] = None


class MeasureGroup(BaseModel):
    id: str
    title: str
    description: Optional[str] = None

    def extract_denom_value_by_id(self, denoms: Optional[list[Denom]]) -> str:
        if denoms is not None and denoms[0].counts is not None:
            for denom_count in denoms[0].counts:
                if self.id == denom_count.groupId:
                    return denom_count.value

        return "0"


class BaselineCharacteristicsModule(BaseModel):
    groups: List[MeasureGroup]
    populationDescription: Optional[str] = None
    typeUnitsAnalyzed: Optional[str] = None
    denoms: Optional[List[Denom]] = None
    measures: List[BaselineMeasure]


class OutcomeCategory(BaseModel):
    title: str
    measurements: List[Measurement]


class MeasureAnalysis(BaseModel):
    paramType: Optional[str] = None
    paramValue: Optional[str] = None
    dispersionType: Optional[AnalysisDispersionType] = None
    dispersionValue: Optional[str] = None
    statisticalMethod: Optional[str] = None
    statisticalComment: Optional[str] = None
    pValue: Optional[str] = None
    pValueComment: Optional[str] = None
    ciNumSides: Optional[ConfidenceIntervalNumSides] = None
    ciPctValue: Optional[str] = None
    ciLowerLimit: Optional[str] = None
    ciUpperLimit: Optional[str] = None
    ciLowerLimitComment: Optional[str] = None
    ciUpperLimitComment: Optional[str] = None
    estimateComment: Optional[str] = None
    testedNonInferiority: Optional[bool] = None
    nonInferiorityType: Optional[NonInferiorityType] = None
    nonInferiorityComment: Optional[str] = None
    otherAnalysisDescription: Optional[str] = None
    groupDescription: Optional[str] = None
    groupIds: Optional[List[str]] = None


class OutcomeMeasure(BaseModel):
    type: OutcomeMeasureType
    title: str
    timeFrame: Optional[str] = None
    description: Optional[str] = None
    populationDescription: Optional[str] = None
    reportingStatus: Optional[ReportingStatus] = None
    anticipatedPostingDate: Optional[str] = None
    paramType: Optional[MeasureParam] = None
    dispersionType: Optional[str] = None
    unitOfMeasure: Optional[str] = None
    calculatePct: Optional[bool] = None
    typeUnitsAnalyzed: Optional[str] = None
    denomUnitsSelected: Optional[str] = None
    groups: Optional[List[MeasureGroup]] = None
    denoms: Optional[List[Denom]] = None
    classes: Optional[List[MeasureClass]] = None
    analyses: Optional[List[MeasureAnalysis]] = None

    def get_group_stats(self, group: MeasureGroup) -> Optional[Measurement]:
        if self.classes is not None:
            for measure_class in self.classes:
                categories = measure_class.categories
                measurements = categories[0].measurements if categories else []

                if measurements is not None:
                    for group_measure in measurements:
                        if group.id == group_measure.groupId:
                            return group_measure

        return None


class OutcomeMeasuresModule(BaseModel):
    outcomeMeasures: List[OutcomeMeasure]


class EventGroup(BaseModel):
    id: str
    title: Optional[str] = None
    description: Optional[str] = None
    deathsNumAffected: Optional[int] = None
    deathsNumAtRisk: Optional[int] = None
    seriousNumAffected: Optional[int] = None
    seriousNumAtRisk: Optional[int] = None
    otherNumAffected: Optional[int] = None
    otherNumAtRisk: Optional[int] = None


class EventStats(BaseModel):
    groupId: str
    numEvents: Optional[int] = None
    numAffected: Optional[int] = None
    numAtRisk: Optional[int] = None


class AdverseEvent(BaseModel):
    term: Optional[str] = None
    organSystem: Optional[str] = None
    sourceVocabulary: Optional[str] = None
    assessmentType: Optional[EventAssessment] = None
    notes: Optional[str] = None
    stats: Optional[List[EventStats]] = None


class AdverseEventsModule(BaseModel):
    frequencyThreshold: Optional[str] = None
    timeFrame: Optional[str] = None
    description: Optional[str] = None
    allCauseMortalityComment: Optional[str] = None
    eventGroups: List[EventGroup]
    seriousEvents: Optional[List[AdverseEvent]] = None
    otherEvents: Optional[List[AdverseEvent]] = None


class LimitationsAndCaveats(BaseModel):
    description: str


class CertainAgreement(BaseModel):
    piSponsorEmployee: bool
    restrictionType: Optional[AgreementRestrictionType] = None
    restrictiveAgreement: Optional[bool] = None
    otherDetails: Optional[str] = None


class PointOfContact(BaseModel):
    title: str
    organization: str
    email: Optional[str] = None
    phone: Optional[str] = None
    phoneExt: Optional[str] = None


class MoreInfoModule(BaseModel):
    limitationsAndCaveats: Optional[LimitationsAndCaveats] = None
    certainAgreement: Optional[CertainAgreement] = None
    pointOfContact: Optional[PointOfContact] = None


class ResultsSection(BaseModel):
    participantFlowModule: ParticipantFlowModule
    baselineCharacteristicsModule: BaselineCharacteristicsModule
    outcomeMeasuresModule: OutcomeMeasuresModule
    adverseEventsModule: Optional[AdverseEventsModule] = None
    moreInfoModule: Optional[MoreInfoModule] = None


class UnpostedEvent(BaseModel):
    type: UnpostedEventType
    date: str
    dateUnknown: Optional[bool] = None


class UnpostedAnnotation(BaseModel):
    unpostedResponsibleParty: Optional[str] = None
    unpostedEvents: Optional[List[UnpostedEvent]]


class ViolationEvent(BaseModel):
    type: ViolationEventType
    description: Optional[str]
    creationDate: Optional[str]
    issuedDate: Optional[str] = None
    releaseDate: Optional[str] = None
    postedDate: Optional[str] = None


class ViolationAnnotation(BaseModel):
    violationEvents: List[ViolationEvent]


class AnnotationModule(BaseModel):
    unpostedAnnotation: Optional[UnpostedAnnotation] = None
    violationAnnotation: Optional[ViolationAnnotation] = None


class AnnotationSection(BaseModel):
    annotationModule: AnnotationModule


class LargeDoc(BaseModel):
    typeAbbrev: Optional[str] = None
    hasProtocol: Optional[bool] = None
    hasSap: Optional[bool] = None
    hasIcf: Optional[bool] = None
    label: Optional[str] = None
    date: Optional[str] = None
    uploadDate: Optional[str] = None
    filename: Optional[str] = None
    size: Optional[int] = None


class LargeDocumentModule(BaseModel):
    noSap: Optional[bool] = None
    largeDocs: Optional[List[LargeDoc]] = None


class DocumentSection(BaseModel):
    largeDocumentModule: Optional[LargeDocumentModule] = None


class FirstMcpInfo(BaseModel):
    postDateStruct: DateStruct


class SubmissionInfo(BaseModel):
    releaseDate: Optional[str] = None
    unreleaseDate: Optional[str] = None
    unreleaseDateUnknown: Optional[bool] = None
    resetDate: Optional[str] = None
    mcpReleaseN: Optional[int] = None


class SubmissionTracking(BaseModel):
    estimatedResultsFirstSubmitDate: Optional[str] = None
    firstMcpInfo: Optional[FirstMcpInfo] = None
    submissionInfos: Optional[List[SubmissionInfo]] = None


class Mesh(BaseModel):
    id: str
    term: str


class BrowseLeaf(BaseModel):
    id: str
    name: str
    asFound: Optional[str] = None
    relevance: Optional[BrowseLeafRelevance] = None


class BrowseBranch(BaseModel):
    abbrev: str
    name: str


class MiscInfoModule(BaseModel):
    versionHolder: str
    removedCountries: Optional[List[str]] = None
    submissionTracking: Optional[SubmissionTracking] = None


class BrowseModule(BaseModel):
    meshes: Optional[List[Mesh]] = None
    ancestors: Optional[List[Mesh]] = None
    browseLeaves: Optional[List[BrowseLeaf]] = None
    browseBranches: Optional[List[BrowseBranch]] = None


class DerivedSection(BaseModel):
    miscInfoModule: Optional[MiscInfoModule] = None
    conditionBrowseModule: Optional[BrowseModule] = None
    interventionBrowseModule: Optional[BrowseModule] = None


class ClinicalTrial(BaseModel):
    protocolSection: ProtocolSection
    resultsSection: Optional[ResultsSection] = None
    annotationSection: Optional[AnnotationSection] = None
    documentSection: Optional[DocumentSection] = None
    derivedSection: DerivedSection
    hasResults: bool

    @classmethod
    def from_json_file(cls, file_path: str) -> "ClinicalTrial":
        with open(file_path, "r") as file:
            data = json.load(file)
        return cls(**data)


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------


def download_clinical_trials(
    data_path: str, test: bool = False, num_processes: Optional[int] = None
) -> None:
    """Download clinical trials data from ClinicalTrials.gov."""
    if num_processes is None:
        num_processes = mp.cpu_count()

    os.makedirs(data_path, exist_ok=True)

    base_params = {
        "format": "json",
        "aggFilters": "studyType:int,results:with,status:com"
        if not test
        else "studyType:int,results:without,status:act",
        "countTotal": "true",
        "pageSize": "1000",  # max page size according to API docs
    }

    # initial request to get total count and first page
    first_page = _fetch_page(None, base_params, test)
    if not first_page:
        return

    total_trials = first_page.get("totalCount", 0)
    logger.info(f"Expected number of trials: {total_trials}")

    # save the first page of trials
    with mp.Pool(processes=num_processes) as pool:
        save_func = partial(_save_trial, data_path)
        pool.map(save_func, first_page["studies"])

    with tqdm(total=total_trials, desc="Downloading trials") as progress_bar:
        progress_bar.update(len(first_page["studies"]))

        # continue with remaining pages
        next_token = first_page.get("nextPageToken")
        while next_token:
            page_data = _fetch_page(next_token, base_params, test)
            if not page_data:
                break

            studies = page_data["studies"]

            with mp.Pool(processes=num_processes) as pool:
                save_func = partial(_save_trial, data_path)
                pool.map(save_func, studies)

            progress_bar.update(len(studies))

            next_token = page_data.get("nextPageToken")


def _fetch_page(
    page_token: Optional[Any], base_params: dict[str, str], test: bool
) -> Optional[dict]:
    """Fetch a single page of clinical trials data."""
    params = base_params.copy()
    if page_token:
        params["pageToken"] = page_token

    response = requests.get(
        "https://clinicaltrials.gov/api/v2/studies",
        params=params,
        headers={"accept": "application/json"},
    )

    if response.status_code != 200:
        logger.error(f"Failed to download trials: {response.text}")
        return None

    return response.json()


def _save_trial(data_path: str, trial: dict[str, Any]) -> str:
    """Save a single trial to a JSON file."""
    nct_id = trial["protocolSection"]["identificationModule"]["nctId"]
    file_path = os.path.join(data_path, f"{nct_id}.json")
    with open(file_path, "w") as f:
        json.dump(trial, f)
    return nct_id
