package main

import (
	"fmt"
	"github.com/jonreiter/govader"
	"io/ioutil"
	"os"
	"regexp"
	"strings"
	"sync"
)

func main() {
	// Map of filenames and correspoinding regular expressions
	filenames_and_regexes := map[string]*regexp.Regexp{
		"OpenWebTextSentiments/African-American.txt":          regexp.MustCompile(`(?i)\bAfrican\WAmerican\b`),
		"OpenWebTextSentiments/Agnostic.txt":                  regexp.MustCompile(`(?i)\bAgnostic\b`),
		"OpenWebTextSentiments/Alabaman.txt":                  regexp.MustCompile(`(?i)\bAlabaman\b`),
		"OpenWebTextSentiments/Alabamian.txt":                 regexp.MustCompile(`(?i)\bAlabamian\b`),
		"OpenWebTextSentiments/Alaskan.txt":                   regexp.MustCompile(`(?i)\bAlaskan\b`),
		"OpenWebTextSentiments/Alt-Right.txt":                 regexp.MustCompile(`(?i)\bAlt\WRight\b`),
		"OpenWebTextSentiments/American_Samoan.txt":           regexp.MustCompile(`(?i)\bAmerican\WSamoan\b`),
		"OpenWebTextSentiments/American-Indian.txt":           regexp.MustCompile(`(?i)\bAmerican\WIndian\b`),
		"OpenWebTextSentiments/Arizonan.txt":                  regexp.MustCompile(`(?i)\bArizonan\b`),
		"OpenWebTextSentiments/Arkansan.txt":                  regexp.MustCompile(`(?i)\bArkansan\b`),
		"OpenWebTextSentiments/Asian.txt":                     regexp.MustCompile(`(?i)\bAsian\b`),
		"OpenWebTextSentiments/Atheist.txt":                   regexp.MustCompile(`(?i)\bAtheist\b`),
		"OpenWebTextSentiments/Black.txt":                     regexp.MustCompile(`(?i)\bBlack\b`),
		"OpenWebTextSentiments/Buddhist.txt":                  regexp.MustCompile(`(?i)\bBuddhist\b`),
		"OpenWebTextSentiments/Californian.txt":               regexp.MustCompile(`(?i)\bCalifornian\b`),
		"OpenWebTextSentiments/Catholic.txt":                  regexp.MustCompile(`(?i)\bCatholic\b`),
		"OpenWebTextSentiments/Caucasian.txt":                 regexp.MustCompile(`(?i)\bCaucasian\b`),
		"OpenWebTextSentiments/Christian.txt":                 regexp.MustCompile(`(?i)\bChristian\b`),
		"OpenWebTextSentiments/Coloradan.txt":                 regexp.MustCompile(`(?i)\bColoradan\b`),
		"OpenWebTextSentiments/Confucian.txt":                 regexp.MustCompile(`(?i)\bConfucian\b`),
		"OpenWebTextSentiments/Connecticuter.txt":             regexp.MustCompile(`(?i)\bConnecticuter\b`),
		"OpenWebTextSentiments/Conservative_Jew.txt":          regexp.MustCompile(`(?i)\bConservative\WJew\b`),
		"OpenWebTextSentiments/Delawarean.txt":                regexp.MustCompile(`(?i)\bDelawarean\b`),
		"OpenWebTextSentiments/Democrat.txt":                  regexp.MustCompile(`(?i)\bDemocrat\b`),
		"OpenWebTextSentiments/Democratic_Socialist.txt":      regexp.MustCompile(`(?i)\bDemocratic\WSocialist\b`),
		"OpenWebTextSentiments/Eastern_Orthodox.txt":          regexp.MustCompile(`(?i)\bEastern\WOrthodox\b`),
		"OpenWebTextSentiments/Floridian.txt":                 regexp.MustCompile(`(?i)\bFloridian\b`),
		"OpenWebTextSentiments/Georgian.txt":                  regexp.MustCompile(`(?i)\bGeorgian\b`),
		"OpenWebTextSentiments/Green.txt":                     regexp.MustCompile(`(?i)\bGreen\b`),
		"OpenWebTextSentiments/Guamanian.txt":                 regexp.MustCompile(`(?i)\bGuamanian\b`),
		"OpenWebTextSentiments/Hawaii_resident.txt":           regexp.MustCompile(`(?i)\bHawaii\Wresident\b`),
		"OpenWebTextSentiments/Hindu.txt":                     regexp.MustCompile(`(?i)\bHindu\b`),
		"OpenWebTextSentiments/Hispanic.txt":                  regexp.MustCompile(`(?i)\bHispanic\b`),
		"OpenWebTextSentiments/Hoosier.txt":                   regexp.MustCompile(`(?i)\bHoosier\b`),
		"OpenWebTextSentiments/Idahoan.txt":                   regexp.MustCompile(`(?i)\bIdahoan\b`),
		"OpenWebTextSentiments/Illinoisan.txt":                regexp.MustCompile(`(?i)\bIllinoisan\b`),
		"OpenWebTextSentiments/Independent.txt":               regexp.MustCompile(`(?i)\bIndependent\b`),
		"OpenWebTextSentiments/Indianian.txt":                 regexp.MustCompile(`(?i)\bIndianian\b`),
		"OpenWebTextSentiments/Iowan.txt":                     regexp.MustCompile(`(?i)\bIowan\b`),
		"OpenWebTextSentiments/Jain.txt":                      regexp.MustCompile(`(?i)\bJain\b`),
		"OpenWebTextSentiments/Jew.txt":                       regexp.MustCompile(`(?i)\bJew\b`),
		"OpenWebTextSentiments/Kansan.txt":                    regexp.MustCompile(`(?i)\bKansan\b`),
		"OpenWebTextSentiments/Kentuckian.txt":                regexp.MustCompile(`(?i)\bKentuckian\b`),
		"OpenWebTextSentiments/LatinX.txt":                    regexp.MustCompile(`(?i)\bLatinX\b`),
		"OpenWebTextSentiments/Libertarian.txt":               regexp.MustCompile(`(?i)\bLibertarian\b`),
		"OpenWebTextSentiments/Louisianian.txt":               regexp.MustCompile(`(?i)\bLouisianian\b`),
		"OpenWebTextSentiments/Mainer.txt":                    regexp.MustCompile(`(?i)\bMainer\b`),
		"OpenWebTextSentiments/Marshallese.txt":               regexp.MustCompile(`(?i)\bMarshallese\b`),
		"OpenWebTextSentiments/Marylander.txt":                regexp.MustCompile(`(?i)\bMarylander\b`),
		"OpenWebTextSentiments/Massachusettsan.txt":           regexp.MustCompile(`(?i)\bMassachusettsan\b`),
		"OpenWebTextSentiments/Michigander.txt":               regexp.MustCompile(`(?i)\bMichigander\b`),
		"OpenWebTextSentiments/Michiganian.txt":               regexp.MustCompile(`(?i)\bMichiganian\b`),
		"OpenWebTextSentiments/Micronesian.txt":               regexp.MustCompile(`(?i)\bMicronesian\b`),
		"OpenWebTextSentiments/Minnesotan.txt":                regexp.MustCompile(`(?i)\bMinnesotan\b`),
		"OpenWebTextSentiments/Mississippian.txt":             regexp.MustCompile(`(?i)\bMississippian\b`),
		"OpenWebTextSentiments/Missourian.txt":                regexp.MustCompile(`(?i)\bMissourian\b`),
		"OpenWebTextSentiments/Montanan.txt":                  regexp.MustCompile(`(?i)\bMontanan\b`),
		"OpenWebTextSentiments/Muslim.txt":                    regexp.MustCompile(`(?i)\bMuslim\b`),
		"OpenWebTextSentiments/Native_American.txt":           regexp.MustCompile(`(?i)\bNative\WAmerican\b`),
		"OpenWebTextSentiments/Nebraskan.txt":                 regexp.MustCompile(`(?i)\bNebraskan\b`),
		"OpenWebTextSentiments/Nevadan.txt":                   regexp.MustCompile(`(?i)\bNevadan\b`),
		"OpenWebTextSentiments/New_Hampshirite.txt":           regexp.MustCompile(`(?i)\bNew\WHampshirite\b`),
		"OpenWebTextSentiments/New_Jerseyan.txt":              regexp.MustCompile(`(?i)\bNew\WJerseyan\b`),
		"OpenWebTextSentiments/New_Mexican.txt":               regexp.MustCompile(`(?i)\bNew\WMexican\b`),
		"OpenWebTextSentiments/New_Yorker.txt":                regexp.MustCompile(`(?i)\bNew\WYorker\b`),
		"OpenWebTextSentiments/North_Carolinian.txt":          regexp.MustCompile(`(?i)\bNorth\WCarolinian\b`),
		"OpenWebTextSentiments/North_Dakotan.txt":             regexp.MustCompile(`(?i)\bNorth\WDakotan\b`),
		"OpenWebTextSentiments/Northern Mariana Islander.txt": regexp.MustCompile(`(?i)\bNorthern\WMariana\WIslander\b`),
		"OpenWebTextSentiments/Ohioan.txt":                    regexp.MustCompile(`(?i)\bOhioan\b`),
		"OpenWebTextSentiments/Oklahoman.txt":                 regexp.MustCompile(`(?i)\bOklahoman\b`),
		"OpenWebTextSentiments/Oregonian.txt":                 regexp.MustCompile(`(?i)\bOregonian\b`),
		"OpenWebTextSentiments/Orthodox_Jew.txt":              regexp.MustCompile(`(?i)\bOrthodox\WJew\b`),
		"OpenWebTextSentiments/Palauan.txt":                   regexp.MustCompile(`(?i)\bPalauan\b`),
		"OpenWebTextSentiments/Pennsylvanian.txt":             regexp.MustCompile(`(?i)\bPennsylvanian\b`),
		"OpenWebTextSentiments/Protestant.txt":                regexp.MustCompile(`(?i)\bProtestant\b`),
		"OpenWebTextSentiments/Puerto_Rican.txt":              regexp.MustCompile(`(?i)\bPuerto\WRican\b`),
		"OpenWebTextSentiments/Reform_Jew.txt":                regexp.MustCompile(`(?i)\bReform\WJew\b`),
		"OpenWebTextSentiments/Republican.txt":                regexp.MustCompile(`(?i)\bRepublican\b`),
		"OpenWebTextSentiments/Rhode_Islander.txt":            regexp.MustCompile(`(?i)\bRhode\WIslander\b`),
		"OpenWebTextSentiments/Samoan.txt":                    regexp.MustCompile(`(?i)\bSamoan\b`),
		"OpenWebTextSentiments/Shi'ite.txt":                   regexp.MustCompile(`(?i)\bShi'ite\b`),
		"OpenWebTextSentiments/Shinto.txt":                    regexp.MustCompile(`(?i)\bShinto\b`),
		"OpenWebTextSentiments/Sikh.txt":                      regexp.MustCompile(`(?i)\bSikh\b`),
		"OpenWebTextSentiments/South_Carolinian.txt":          regexp.MustCompile(`(?i)\bSouth\WCarolinian\b`),
		"OpenWebTextSentiments/South_Dakotan.txt":             regexp.MustCompile(`(?i)\bSouth\WDakotan\b`),
		"OpenWebTextSentiments/Sunni.txt":                     regexp.MustCompile(`(?i)\bSunni\b`),
		"OpenWebTextSentiments/Taoist.txt":                    regexp.MustCompile(`(?i)\bTaoist\b`),
		"OpenWebTextSentiments/Tennessean.txt":                regexp.MustCompile(`(?i)\bTennessean\b`),
		"OpenWebTextSentiments/Texan.txt":                     regexp.MustCompile(`(?i)\bTexan\b`),
		"OpenWebTextSentiments/Today.txt":                     regexp.MustCompile(`(?i)\bToday\b`),
		"OpenWebTextSentiments/Utahn.txt":                     regexp.MustCompile(`(?i)\bUtahn\b`),
		"OpenWebTextSentiments/Vermonter.txt":                 regexp.MustCompile(`(?i)\bVermonter\b`),
		"OpenWebTextSentiments/Virgin_Islander.txt":           regexp.MustCompile(`(?i)\bVirgin\WIslander\b`),
		"OpenWebTextSentiments/Virginian.txt":                 regexp.MustCompile(`(?i)\bVirginian\b`),
		"OpenWebTextSentiments/Washingtonian.txt":             regexp.MustCompile(`(?i)\bWashingtonian\b`),
		"OpenWebTextSentiments/West_Virginian.txt":            regexp.MustCompile(`(?i)\bWest\WVirginian\b`),
		"OpenWebTextSentiments/White.txt":                     regexp.MustCompile(`(?i)\bWhite\b`),
		"OpenWebTextSentiments/Wisconsinite.txt":              regexp.MustCompile(`(?i)\bWisconsinite\b`),
		"OpenWebTextSentiments/Wyomingite.txt":                regexp.MustCompile(`(?i)\bWyomingite\b`),
		"OpenWebTextSentiments/bisexual.txt":                  regexp.MustCompile(`(?i)\bbisexual\b`),
		"OpenWebTextSentiments/centrist.txt":                  regexp.MustCompile(`(?i)\bcentrist\b`),
		"OpenWebTextSentiments/conservative.txt":              regexp.MustCompile(`(?i)\bconservative\b`),
		"OpenWebTextSentiments/gay.txt":                       regexp.MustCompile(`(?i)\bgay\b`),
		"OpenWebTextSentiments/heterosexual.txt":              regexp.MustCompile(`(?i)\bheterosexual\b`),
		"OpenWebTextSentiments/homosexual.txt":                regexp.MustCompile(`(?i)\bhomosexual\b`),
		"OpenWebTextSentiments/lesbian.txt":                   regexp.MustCompile(`(?i)\blesbian\b`),
		"OpenWebTextSentiments/man.txt":                       regexp.MustCompile(`(?i)\bman\b`),
		"OpenWebTextSentiments/nonbinary.txt":                 regexp.MustCompile(`(?i)\bnonbinary\b`),
		"OpenWebTextSentiments/progressive.txt":               regexp.MustCompile(`(?i)\bprogressive\b`),
		"OpenWebTextSentiments/queer.txt":                     regexp.MustCompile(`(?i)\bqueer\b`),
		"OpenWebTextSentiments/straight.txt":                  regexp.MustCompile(`(?i)\bstraight\b`),
		"OpenWebTextSentiments/transgender man.txt":           regexp.MustCompile(`(?i)\btransgender\Wman\b`),
		"OpenWebTextSentiments/transgender_woman.txt":         regexp.MustCompile(`(?i)\btransgender\Wwoman\b`),
		"OpenWebTextSentiments/transgender.txt":               regexp.MustCompile(`(?i)\btransgender\b`),
		"OpenWebTextSentiments/woman.txt":                     regexp.MustCompile(`(?i)\bwoman\b`),
	}

	// Read file
	filename := os.Args[1]
	content, err := ioutil.ReadFile(filename)
	if err != nil {
		fmt.Println(err)
	}

	// Remove null characters from text and split by regular expression
	texts := strings.Replace(string(content), "\x00", "", -1)
	fmt.Println(texts)
	splitter := regexp.MustCompile(`\d{7}-[0-9a-f]{32}\.txt\d{49} \dustar  \d{14}`)
	split_texts := splitter.Split(texts, -1)
	split_texts_length := len(split_texts)

	analyzer := govader.NewSentimentIntensityAnalyzer() // VADER sentiment analyzer

	var wait_group sync.WaitGroup
	wait_group.Add(split_texts_length)

	for i := 0; i < split_texts_length; i++ {
		go func(i int) {
			defer wait_group.Done()
			text := split_texts[i]
			if text != "" {
				sentiment := analyzer.PolarityScores(string(text)).Compound // Get VADER score for text
				write_matches([]byte(text), sentiment, filenames_and_regexes)
			}
		}(i)
	}
	wait_group.Wait()
}

func write_matches(text []byte, sentiment float64, regexmap map[string]*regexp.Regexp) {
	// Append sentiment to all.txt
	append_to_file("OpenWebTextSentiments/all.txt", sentiment)

	// Write sentiment to regex matches
	for filename, regex := range regexmap { // Iterate through map of filenames and regexes
		if regex.Match(text) { // If there's a regex match in text, append sentiment to relevant file
			append_to_file(filename, sentiment)
		}
	}
}

func append_to_file(filename string, sentiment float64) {
	f, err := os.OpenFile(filename, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		fmt.Println(err)
	}
	_, _ = f.Write([]byte(fmt.Sprintf("%f\n", sentiment)))
	f.Close()
}
