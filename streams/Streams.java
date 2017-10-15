package streams;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class Streams {
	
	private static boolean compararPalabras(String a, String b) {
		if(a.length()>b.length() || a.equals(b)) return false;
		char[] charsA = a.toCharArray();
		char[] charsB = b.toCharArray();
		int tamañoA = a.length();
		int i = 0;
		for(char letra : charsB) {
			if(i < tamañoA ) {
				if(letra == charsA[i]) i++;
			}else {
				return true;
			}
		}
		return (i == tamañoA) ? true : false;
	}
	
	private static boolean esVocal(char letra) {
		return (letra == 'a' ||
				letra == 'e' ||
				letra == 'i' ||
				letra == 'o' ||
				letra == 'u')? true : false;
	}
	
	private static boolean mismasVocalesAux(String a, String b){
		if(a.length() > b.length()) {
			String c = b;
			b = a;
			a = c;
		}
		for(char letra : a.toCharArray()) {
			if(esVocal(letra)) {
				if(!b.contains(letra + "")) return false;
			}
		}
		return true;
	}

	// Cuente cuantas palabras, tiene los mismos charas en otras palabras:
	// ejemplo casa esta contenida en carrosa
	private static void contieneCaracteres(List<String> wordsList) {
		Optional.ofNullable(wordsList).ifPresent(words ->{ 
			int cantidad = words.parallelStream()
				.filter(word1 -> {
					return words.parallelStream()
						.anyMatch( word2 -> compararPalabras(word1, word2));
			}).collect(Collectors.toList()).size();
			System.out.println("Palabras que estan contenidas en otra: " + cantidad);
		});
	}
	
	private static void palabrasDentroDeOtras(List<String> wordsList) {
		// Cuente cuantas palabras, son parte de otra palabra:
		// ejemplo casa esta contenida en carcasa

		Optional.ofNullable(wordsList).ifPresent(words ->{ 
			int num = words.parallelStream()
			// para cada palabra en "words"
			.filter( word -> {
				return words
					.parallelStream()
					// busca si "word" esta contenida en alguna palabra
					.anyMatch(word2 -> word2.contains(word) && word != word2);
			})
			// crea una lista de las coincidencias y obtiene el tamaño
			.collect(Collectors.toList()).size();
			System.out.println("Palabras que son parte de otra palabra: " + num);
		});
	}
	
	private static void mismasVocales(List<String> wordsList) {
		// Cuente cuantas palabras, son parte de otra palabra:
		// ejemplo casa esta contenida en carcasa

		Optional.ofNullable(wordsList).ifPresent(words ->{ 
			int num = words.parallelStream()
			// para cada palabra en "words"
			.filter( word -> {
				return words
					.parallelStream()
					// busca si "word" esta contenida en alguna palabra
					.anyMatch(word2 -> word != word2 && mismasVocalesAux(word,word2) );
			})
			// crea una lista de las coincidencias y obtiene el tamaño
			.collect(Collectors.toList()).size();
			System.out.println("Palabras que tienen las mismas vocales: " + num);
		});
	}
	
	public static void main(String[] args) {
		List<String> words = null;
		String archivo = "words.txt";
		try (Stream<String> stream = Files.lines(Paths.get(archivo))) {
            words = stream.collect(Collectors.toList());            
        } catch (IOException e) {	
        		System.out.println("No se pudo leer el archivo");
        }
		contieneCaracteres(words);
		palabrasDentroDeOtras(words);
		mismasVocales(words);
		
	}

}
