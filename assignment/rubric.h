#define RUBRIC_CPU 0
#define RUBRIC_GPU 1
#define RUBRIC_MPI 2
#define RUBRIC_LENGTH 8

float rubric[3][RUBRIC_LENGTH];

int rubricInit()
{
    //CPU MARKS
    rubric[RUBRIC_CPU][7] = 2.1;
    rubric[RUBRIC_CPU][6] = 19.4;
    rubric[RUBRIC_CPU][5] = 29.9;
    rubric[RUBRIC_CPU][4] = 58.8;
    rubric[RUBRIC_CPU][3] = 129.2;
    rubric[RUBRIC_CPU][2] = 1000;
    // wrong answer, or doesn't compile
    rubric[RUBRIC_CPU][1] = 1000;
    rubric[RUBRIC_CPU][0] = 1000;

    //GPU MARKS
    rubric[RUBRIC_GPU][7] = 4.1;
    rubric[RUBRIC_GPU][6] = 6.2;
    rubric[RUBRIC_GPU][5] = 19.9;
    rubric[RUBRIC_GPU][4] = 34.2;
    rubric[RUBRIC_GPU][3] = 100;
    rubric[RUBRIC_GPU][2] = 1000;
    // wrong answer, or doesn't compile
    rubric[RUBRIC_GPU][1] = 1000;
    rubric[RUBRIC_GPU][0] = 1000;

    //MPI MARKS
    rubric[RUBRIC_MPI][7] = 4.3;
    rubric[RUBRIC_MPI][6] = 9.9;
    rubric[RUBRIC_MPI][5] = 15.4;
    rubric[RUBRIC_MPI][4] = 30.3;
    rubric[RUBRIC_MPI][3] = 64.8;
    rubric[RUBRIC_MPI][2] = 1000;
    // wrong answer, or doesn't compile
    rubric[RUBRIC_MPI][1] = 1000;
    rubric[RUBRIC_MPI][0] = 1000;  
    return 1;
}

float getGrade(float performanceFactor, float err, float* gradeTable)
{
    //floating point error tolerance of the answer given
    const float errTolerance = 1e-7;

    // Matrix multiplication doesn't work properly. 1 point for submitting something that runs at least.
    if (err > errTolerance)
    {
        return 1;
    }
    else
    {
        // Matrix multiplication works, but is about as slow as possible.
        if (performanceFactor >= gradeTable[2])
        {
            return 2;
        }
        else
        {
            // God-like performance. Full marks.
            if (performanceFactor < gradeTable[RUBRIC_LENGTH - 1])
            {
                return RUBRIC_LENGTH - 1;
            }
            else
            {
                for (int gradeIdx = RUBRIC_LENGTH; gradeIdx >= 1; gradeIdx--)
                {
                    // Linearly interpolate between the levels in the grade table to assign a grade
                    if (performanceFactor > gradeTable[gradeIdx] && performanceFactor <= gradeTable[gradeIdx - 1])
                    {
                        float x1 = gradeTable[gradeIdx];
                        float x2 = gradeTable[gradeIdx - 1];
                        float y1 = gradeIdx;
                        float y2 = gradeIdx - 1;
                        float x = performanceFactor;

                        float grade = y1 + ((x - x1) / (x2 - x1) * (y2 - y1));
                        grade = ceil(grade * 10.0);
                        grade = grade / 10.0;
                        return grade;
                    }
                }
            }
        }
    }
    return 1;
}