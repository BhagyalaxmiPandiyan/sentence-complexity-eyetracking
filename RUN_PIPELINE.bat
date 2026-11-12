@echo off
rem ===========================================================================
rem  Double-click to re-run the pipeline steps and see the thesis numbers.
rem  Each option prints its results in this window. Nothing in the thesis or the
rem  toolkit is changed (results files are re-created with the same numbers).
rem ===========================================================================
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PY=C:\Python313\python.exe
set ROOT=%~dp0
rem Optional: run one option directly, e.g.  RUN_PIPELINE.bat 2
set ARG=%~1
if not "%ARG%"=="" (set choice=%ARG%& goto dispatch)

:menu
cls
echo ============================================================================
echo   Measuring Sentence Complexity - re-run the pipeline
echo   Bhagyalaxmi Pandiyan (M25DE2070) - IIT Jodhpur
echo ============================================================================
echo.
echo   LIVE DEMO
echo     1. Start the toolkit (opens in the browser)
echo.
echo   PHASE 1 - English replication of Singh et al. (2016)
echo     2. System 1: reading-time models, ablation + Table 3        (~10 seconds)
echo     3. System 2: which sentence is harder, 10-fold CV           (30-40 minutes)
echo.
echo   PHASE 2 - GPT-2 surprisal
echo     4. System 1 with and without GPT-2 surprisal                (~15 seconds)
echo.
echo   PHASE 3 - Tamil
echo     5. Word vs morphology metric on StoryWeaver                 (~15 seconds)
echo     6. Webcam eye-tracking study (5 readers)                    (~4 minutes)
echo     7. External test on OasisSimp (1,904 human pairs)           (~20 seconds)
echo.
echo   DOCUMENTS
echo     8. Open the thesis (PDF)
echo     9. Open the step-by-step replication guide
echo.
echo     0. Exit
echo.
set choice=
set /p choice=  Type a number and press Enter: 
if "%choice%"=="" goto menu

:dispatch
if "%choice%"=="1" start "" "%ROOT%START_TOOLKIT.bat" & goto menu
if "%choice%"=="2" goto p1s1
if "%choice%"=="3" goto p1s2
if "%choice%"=="4" goto p2
if "%choice%"=="5" goto p3a
if "%choice%"=="6" goto p3et
if "%choice%"=="7" goto p3ext
if "%choice%"=="8" start "" "%ROOT%thesis_report\MTP_Thesis_Bhagyalaxmi_Pandiyan_M25DE2070.pdf" & goto menu
if "%choice%"=="9" start "" notepad "%ROOT%REPLICATION.md" & goto menu
if "%choice%"=="0" exit /b
goto menu

:p1s1
cd /d "%ROOT%phase1_replication"
echo.
echo  Phase 1 - step 4: System 1 (Ridge regression, Dundee corpus)
echo  Expected test R2: FFD 0.444, FPD 0.558, RPD 0.555, TD 0.536
echo  (Thesis Table 4.1 shows 0.447 / 0.561 / 0.558 / 0.538 from an earlier feature
echo   version; the 0.003 difference is explained in Section 4 of the thesis.)
echo.
rem keep the saved models and predictions exactly as they are
copy /y output\system1_models.pkl "%TEMP%\system1_models.bak" >nul
copy /y output\system1_predictions.csv "%TEMP%\system1_predictions.bak" >nul
%PY% src\step4_system1.py
copy /y "%TEMP%\system1_models.bak" output\system1_models.pkl >nul
copy /y "%TEMP%\system1_predictions.bak" output\system1_predictions.csv >nul
goto done

:p1s2
cd /d "%ROOT%phase1_replication"
echo.
echo  Phase 1 - step 5: System 2 on 117,712 Wikipedia / Simple Wikipedia pairs
echo  Expected accuracy: Base RT 74.15%%, Extended RT 75.26%%  (paper 73.82%% / 75.21%%)
echo  This takes 30-40 minutes.
echo.
%PY% src\step5_system2.py
goto done

:p2
cd /d "%ROOT%phase1_replication"
echo.
echo  Phase 2: System 1 with GPT-2 surprisal (3 sentences of context)
echo  Expected R2 with GPT-2: FFD 0.458, FPD 0.583, RPD 0.582, TD 0.568
echo.
%PY% experiments\train_system1_gpt2.py --surprisal-file output\gpt2_surprisal_gpt2_ctx.csv
goto done

:p3a
cd /d "%ROOT%phase3_tamil_extension"
echo.
echo  Phase 3 - step 6: word metric vs morphology metric (StoryWeaver test pairs)
echo  Expected: word-level 63.8%%, morphology 60.5%%, word + morphology 63.4%%
echo.
%PY% src\step6_evaluate_metric.py storyweaver
goto done

:p3et
cd /d "%ROOT%phase3_tamil_extension"
echo.
echo  Phase 3 - step 11: webcam eye-tracking (System 1 on Tamil reading times)
echo  Expected TD R2: controls 0.027, word 0.103, morphology 0.071, word + morphology 0.099
echo.
%PY% src\step11_eyetracking.py
goto done

:p3ext
cd /d "%ROOT%phase3_tamil_extension"
echo.
echo  Phase 3 - step 12: OasisSimp external test (human simplifications)
echo  Expected: word metric 95.6%%, morph metric 93.4%%
echo.
%PY% src\step12_oasissimp.py
goto done

:done
echo.
echo ============================================================================
if not "%ARG%"=="" exit /b
echo  Finished. Press any key to go back to the menu.
pause >nul
goto menu
