$instanceSizeTestFileNames = @("instanceSizeTest1","instanceSizeTest2","instanceSizeTest3","instanceSizeTest4","instanceSizeTest5","instanceSizeTest6","instanceSizeTest7","instanceSizeTest8")
$distanceTestFileNames = @("maxDistTest1","maxDistTest2","maxDistTest3","maxDistTest4")
$prevalenceTestFileNames = @("minPrevTest1","minPrevTest2")

foreach ($instanceSizeTestFile in $instanceSizetestFileNames)
{
  for($i = 1; $i -le 20; $i++)
  {
    .\GPUPatternMining.CPU.App.exe $instanceSizeTestFile 3 0.2 $i
    .\GPUPatternMining.Gpu.App.exe $instanceSizeTestFile 3 0.2 $i
  }
}

foreach ($distanceTestFile in $distanceTestFileNames)
{
  for($dist = 3; $dist -le 10; $dist++)
  {
    for($i = 1; $i -le 20; $i++)
    {
      .\GPUPatternMining.CPU.App.exe $distanceTestFile $dist 0.2 $i
      .\GPUPatternMining.Gpu.App.exe $distanceTestFile $dist 0.2 $i
    }
  }
}

foreach ($prevalenceTestFile in $prevalenceTestFileNames)
{
  for($prev = 0.05; $prev -le 0.5; $prev = $prev + 0.05)
  {
    for($i = 1; $i -le 20; $i++)
    {
      .\GPUPatternMining.CPU.App.exe $prevalenceTestFile 5 $prev $i
      .\GPUPatternMining.Gpu.App.exe $prevalenceTestFile 5 $prev $i
    }
  }
}
