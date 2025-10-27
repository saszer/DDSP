@echo off
echo Checking if trained model synthesis was used...
echo.
python test_midi_upload.py > test_log.txt 2>&1
echo.
findstr /C:"Trained synthesis OK" /C:"Trained synthesis failed" /C:"Falling back" test_log.txt
echo.
echo Test complete. Full log saved to test_log.txt

